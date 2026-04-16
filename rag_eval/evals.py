import sys
import csv
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI

from ragas import experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from search import build_chain

HF_DATASET = "yannelli/laravel-11-qa"
MAX_QUESTIONS = 200
RANDOM_SEED = 42
MODEL_NAME = "llama3.1"

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

qa_chain = build_chain()
llm = llm_factory(MODEL_NAME, provider="openai", client=client)


def load_eval_rows():
    """
    Load and sample exactly MAX_QUESTIONS rows from yannelli/laravel-11-qa.
    """
    hf = load_dataset(HF_DATASET)

    split_name = "train" if "train" in hf else list(hf.keys())[0]
    split = hf[split_name]

    if len(split) < MAX_QUESTIONS:
        raise ValueError(
            f"Dataset only has {len(split)} rows, expected at least {MAX_QUESTIONS}."
        )

    sampled = split.shuffle(seed=RANDOM_SEED).select(range(MAX_QUESTIONS))

    rows = []
    for row in sampled:
        question = row.get("question") or row.get("human")
        answer = row.get("ground_truth") or row.get("answer") or row.get("assistant")
        rows.append(
            {
                "question": str(question or "").strip(),
                "grading_notes": str(answer).strip(),
            }
        )

    rows = [r for r in rows if r["question"]]
    if len(rows) != MAX_QUESTIONS:
        raise ValueError(
            f"Prepared {len(rows)} valid rows after cleanup, expected {MAX_QUESTIONS}."
        )

    return rows


my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)


def fallback_judge(response_text: str, grading_notes: str) -> str:
    """
    Fallback scorer for local models when RAGAS DiscreteMetric parsing fails.
    Forces a strict single-token output: pass|fail.
    """
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator. Return only one word: pass or fail. "
                    "Do not return JSON, explanations, or extra text."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Compare RESPONSE to GRADING_NOTES. "
                    "Return pass only if response is materially correct.\n\n"
                    f"RESPONSE:\n{response_text}\n\n"
                    f"GRADING_NOTES:\n{grading_notes}"
                ),
            },
        ],
    )
    raw = (completion.choices[0].message.content or "").strip().lower()
    return "pass" if raw.startswith("pass") else "fail"


@experiment()
async def run_experiment(row):
    answer = qa_chain.invoke(row["question"])

    score_source = "ragas_discrete_metric"
    try:
        score = my_metric.score(
            llm=llm,
            response=answer or " ",
            grading_notes=row["grading_notes"],
        )
        score_value = score.value
    except Exception:
        score_source = "fallback_judge"
        score_value = fallback_judge(answer or " ", row["grading_notes"])

    experiment_view = {
        **row,
        "response": answer,
        "score": score_value,
        "log_file": score_source,
    }
    return experiment_view


async def main():
    eval_rows = load_eval_rows()
    print(f"Loaded {len(eval_rows)} evaluation questions from {HF_DATASET}.")
    results = []
    for idx, row in enumerate(eval_rows, 1):
        results.append(await run_experiment(row))
        if idx % 20 == 0 or idx == len(eval_rows):
            print(f"Scored {idx}/{len(eval_rows)}")

    print("Experiment completed successfully!")

    out_dir = Path(".") / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "laravel11_qa_eval_200.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "grading_notes", "response", "score", "log_file"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
