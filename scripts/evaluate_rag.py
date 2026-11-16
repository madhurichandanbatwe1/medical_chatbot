"""Evaluate the medical RAG pipeline against a CSV question/answer dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import statistics
import sys
import unicodedata
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path.home() / "Downloads" / "qa_dataset.csv"
REQUIRED_COLUMNS = {
    "id",
    "entry",
    "question",
    "reference_answer",
    "key_facts",
    "source",
}
JUDGE_SCORES = (
    "correctness",
    "answer_relevance",
    "faithfulness",
    "context_relevance",
)
FACT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
RESULT_COLUMNS = (
    "id",
    "entry",
    "question",
    "reference_answer",
    "key_facts",
    "answer",
    "context_key_fact_recall",
    "answer_key_fact_recall",
    "reference_token_precision",
    "reference_token_recall",
    "reference_token_f1",
    "retrieved_entry_metadata_coverage",
    "retrieved_entry_match_rate",
    "retrieved_sources",
    "retrieved_contexts",
    "generation_error",
    "judge_correctness",
    "judge_answer_relevance",
    "judge_faithfulness",
    "judge_context_relevance",
    "judge_notes",
    "judge_error",
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized).split())


def split_key_facts(value: str) -> list[str]:
    return [fact.strip() for fact in value.split("|") if fact.strip()]


def key_fact_recall(facts: list[str], text: str) -> float | None:
    """Count facts covered by at least 80% of their normalized content words."""
    if not facts:
        return None

    text_tokens = set(normalize_text(text).split())
    covered = 0
    for fact in facts:
        fact_tokens = set(normalize_text(fact).split()) - FACT_STOPWORDS
        if not fact_tokens:
            fact_tokens = set(normalize_text(fact).split())
        if fact_tokens and len(fact_tokens & text_tokens) / len(fact_tokens) >= 0.8:
            covered += 1
    return covered / len(facts)


def reference_token_scores(reference: str, answer: str) -> tuple[float, float, float]:
    reference_tokens = Counter(normalize_text(reference).split())
    answer_tokens = Counter(normalize_text(answer).split())
    if not reference_tokens and not answer_tokens:
        return 1.0, 1.0, 1.0
    if not reference_tokens or not answer_tokens:
        return 0.0, 0.0, 0.0

    overlap = sum((reference_tokens & answer_tokens).values())
    precision = overlap / sum(answer_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def entry_metadata_scores(
    documents: list[Any],
    expected_entry: str,
) -> tuple[float | None, float | None]:
    if not documents:
        return None, None

    entries = [
        str(document.metadata.get("entry") or document.metadata.get("entry_name")).strip()
        for document in documents
        if document.metadata.get("entry") or document.metadata.get("entry_name")
    ]
    coverage = len(entries) / len(documents)
    if not entries:
        return coverage, None

    matching = sum(entry.casefold() == expected_entry.strip().casefold() for entry in entries)
    return coverage, matching / len(entries)


def _document_records(documents: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "metadata": document.metadata,
            "page_content": document.page_content,
        }
        for document in documents
    ]


def format_judge_context(records: list[dict[str, Any]]) -> str:
    if not records:
        return "(no retrieved chunks)"

    chunks = []
    for index, record in enumerate(records, start=1):
        metadata = json.dumps(
            record.get("metadata", {}),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        chunks.append(
            f"[CHUNK {index}; METADATA={metadata}]\n"
            f"{record.get('page_content', '')}"
        )
    return "\n\n".join(chunks)


def evaluate_rows(
    rows: list[dict[str, str]],
    run_query: Callable[[str], dict[str, Any]],
    judge: Any | None = None,
) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        response = run_query(row["question"])
        answer = response.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise TypeError(f"Answer for {row['id']} must be a string or None")
        documents = response.get("context")
        if not isinstance(documents, list):
            raise TypeError(f"Context for {row['id']} must be a list of documents")

        facts = split_key_facts(row["key_facts"])
        answer_text = answer or ""
        context_text = "\n\n".join(document.page_content for document in documents)
        document_records = _document_records(documents)
        precision, recall, f1 = (
            reference_token_scores(row["reference_answer"], answer_text)
            if answer is not None
            else (None, None, None)
        )
        entry_coverage, entry_match_rate = entry_metadata_scores(
            documents,
            row["entry"],
        )

        generation_error = response.get("generation_error", "")
        judge_result: dict[str, Any] = {}
        judge_error = ""
        if judge is not None and answer is not None:
            try:
                judge_result = score_with_local_judge(
                    judge=judge,
                    question=row["question"],
                    expected_entry=row["entry"],
                    reference_answer=row["reference_answer"],
                    key_facts=facts,
                    answer=answer,
                    context=format_judge_context(document_records),
                )
            except (TypeError, ValueError) as error:
                judge_error = f"{type(error).__name__}: {error}"

        result: dict[str, Any] = {
            "id": row["id"],
            "entry": row["entry"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "key_facts": row["key_facts"],
            "answer": answer_text,
            "context_key_fact_recall": key_fact_recall(facts, context_text),
            "answer_key_fact_recall": (
                key_fact_recall(facts, answer_text) if answer is not None else None
            ),
            "reference_token_precision": precision,
            "reference_token_recall": recall,
            "reference_token_f1": f1,
            "retrieved_entry_metadata_coverage": entry_coverage,
            "retrieved_entry_match_rate": entry_match_rate,
            "retrieved_sources": json.dumps(
                [document.metadata for document in documents],
                ensure_ascii=False,
                default=str,
            ),
            "retrieved_contexts": json.dumps(
                document_records,
                ensure_ascii=False,
                default=str,
            ),
            "generation_error": generation_error,
            "judge_notes": judge_result.get("notes", ""),
            "judge_error": judge_error,
        }
        for score_name in JUDGE_SCORES:
            result[f"judge_{score_name}"] = judge_result.get(score_name)
        results.append(result)
    return results


def score_with_local_judge(
    judge: Any,
    question: str,
    expected_entry: str,
    reference_answer: str,
    key_facts: list[str],
    answer: str,
    context: str,
) -> dict[str, Any]:
    prompt = f"""Evaluate this medical RAG response against the supplied question,
reference, key facts, and retrieved chunks. Use only the retrieved chunks to judge
faithfulness; do not use outside medical knowledge.

Use only these score values: 0.0, 0.25, 0.5, 0.75, or 1.0.
- correctness: 1.0 only when the answer correctly addresses every part asked and
  includes the applicable required facts. Lower the score for omissions or errors.
  Do not penalize omission of facts the question did not ask for.
- answer_relevance: 1.0 when the answer directly addresses the question without
  distracting unrelated material; lower it for missing parts or irrelevant detail.
- faithfulness: 1.0 only when every substantive medical claim is supported by a
  retrieved chunk. Lower it for any unsupported or contradicted claim. If no chunk
  supports a claim, treat it as unsupported even if it is medically plausible.
- context_relevance: score the retrieved chunks as a set for the expected entry
  and question. 1.0 means all or nearly all chunks are relevant; 0.5 means a
  substantial mixture of relevant and unrelated chunks; 0.0 means none are relevant.

Return only a JSON object containing numeric values for those four scores and a
short "notes" string. In notes, identify the main omitted, unsupported, or
irrelevant evidence, citing chunk numbers when possible. Do not provide hidden
chain-of-thought.

Question: {question}
Expected entry: {expected_entry}
Reference answer: {reference_answer}
Key facts: {json.dumps(key_facts, ensure_ascii=False)}
Retrieved chunks:
{context}
Chatbot answer: {answer}
"""
    response = judge.invoke(prompt)
    content = response.content
    if not isinstance(content, str):
        raise TypeError("The local Ollama judge must return text")
    try:
        scores = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("The local Ollama judge returned invalid JSON") from error

    if not isinstance(scores, dict):
        raise ValueError("The local Ollama judge response must be a JSON object")
    for score_name in JUDGE_SCORES:
        score = scores.get(score_name)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"The local Ollama judge omitted numeric {score_name}")
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"The local Ollama judge returned an invalid {score_name}")
        scores[score_name] = float(score)
    notes = scores.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("The local Ollama judge notes must be a string")
    scores["notes"] = notes
    return scores


def rejudge_existing_results(
    results_path: Path,
    judge: Any,
) -> list[dict[str, Any]]:
    with results_path.open("r", encoding="utf-8-sig", newline="") as result_file:
        reader = csv.DictReader(result_file)
        required = {
            "id",
            "entry",
            "question",
            "reference_answer",
            "key_facts",
            "answer",
            "retrieved_contexts",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Existing results are missing required columns: "
                + ", ".join(sorted(missing))
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"No result rows found in {results_path}")

    results = []
    for row in rows:
        result: dict[str, Any] = {column: row.get(column, "") for column in RESULT_COLUMNS}
        result["judge_error"] = ""
        result["judge_notes"] = ""
        for score_name in JUDGE_SCORES:
            result[f"judge_{score_name}"] = None

        for metric_name in (
            "context_key_fact_recall",
            "answer_key_fact_recall",
            "reference_token_precision",
            "reference_token_recall",
            "reference_token_f1",
            "retrieved_entry_metadata_coverage",
            "retrieved_entry_match_rate",
        ):
            value = result.get(metric_name)
            result[metric_name] = float(value) if value not in (None, "") else None

        if row.get("generation_error") or not row.get("answer", "").strip():
            results.append(result)
            continue

        try:
            records = json.loads(row["retrieved_contexts"])
            if not isinstance(records, list):
                raise ValueError("Retrieved contexts must be a JSON list")
            score = score_with_local_judge(
                judge=judge,
                question=row["question"],
                expected_entry=row["entry"],
                reference_answer=row["reference_answer"],
                key_facts=split_key_facts(row["key_facts"]),
                answer=row["answer"],
                context=format_judge_context(records),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            result["judge_error"] = f"{type(error).__name__}: {error}"
        else:
            result["judge_notes"] = score.get("notes", "")
            for score_name in JUDGE_SCORES:
                result[f"judge_{score_name}"] = score[score_name]
        results.append(result)
    return results


def load_dataset(dataset_path: Path) -> list[dict[str, str]]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        rows = [
            {column: (row.get(column) or "").strip() for column in REQUIRED_COLUMNS}
            for row in reader
        ]

    if not rows:
        raise ValueError(f"No evaluation examples found in {dataset_path}")
    for row in rows:
        if not row["id"] or not row["question"]:
            raise ValueError("Each dataset row must have a non-empty id and question")
    return rows


def build_rag_runner(retrieval_only: bool) -> Callable[[str], dict[str, Any]]:
    if retrieval_only:
        from sentence_transformers import CrossEncoder

        from app.components.retriever import RERANKER_MODEL, retrieve_and_rerank
        from app.components.vector_store import load_vector_store

        vector_store = load_vector_store()
        if vector_store is None:
            raise RuntimeError("The configured FAISS vector store is not available")
        reranker = CrossEncoder(RERANKER_MODEL)

        def retrieve(question: str) -> dict[str, Any]:
            return {
                "answer": None,
                "context": retrieve_and_rerank(question, vector_store, reranker),
            }

        return retrieve

    from app.components.retriever import create_qa_chain

    chain = create_qa_chain()

    def run_chain(question: str) -> dict[str, Any]:
        response = chain.invoke({"input": question})
        if not isinstance(response, dict):
            raise TypeError("The RAG chain must return a dictionary")
        answer = response.get("answer")
        context = response.get("context")
        if not isinstance(context, list):
            raise TypeError("The RAG chain did not return its retrieved documents")
        if not isinstance(answer, str) or not answer.strip():
            return {
                "answer": None,
                "context": context,
                "generation_error": "The RAG chain returned an empty or invalid answer",
            }
        return {"answer": answer, "context": context}

    return run_chain


def build_local_judge(model: str) -> Any:
    from app.config.config import OLLAMA_BASE_URL
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        reasoning=False,
        format="json",
        num_ctx=8192,
        num_predict=512,
    )


def configure_langsmith_tracing(enabled: bool, project: str | None = None) -> None:
    if not enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return

    if not (
        os.environ.get("LANGSMITH_API_KEY")
        or os.environ.get("LANGCHAIN_API_KEY")
    ):
        raise RuntimeError(
            "LangSmith tracing requires LANGSMITH_API_KEY in the environment or .env"
        )

    project_name = project or os.environ.get(
        "LANGSMITH_PROJECT",
        "medical-chatbot-rag-eval",
    )
    os.environ["LANGSMITH_PROJECT"] = project_name
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "context_key_fact_recall",
        "answer_key_fact_recall",
        "reference_token_f1",
        "retrieved_entry_metadata_coverage",
        "retrieved_entry_match_rate",
        *(f"judge_{score_name}" for score_name in JUDGE_SCORES),
    )
    summary: dict[str, Any] = {"examples": len(results)}
    generation_errors = [
        result["id"] for result in results if result.get("generation_error")
    ]
    judge_errors = [result["id"] for result in results if result.get("judge_error")]
    summary["generation_error_count"] = len(generation_errors)
    summary["generation_error_ids"] = generation_errors
    summary["judge_error_count"] = len(judge_errors)
    summary["judge_error_ids"] = judge_errors
    for metric_name in metric_names:
        values = [
            result[metric_name]
            for result in results
            if isinstance(result.get(metric_name), (int, float))
            and not isinstance(result.get(metric_name), bool)
        ]
        summary[metric_name] = (
            round(statistics.mean(values), 4) if values else None
        )
    return summary


def write_results(
    results: list[dict[str, Any]],
    output_path: Path,
    dataset_path: Path,
    retrieval_only: bool,
    judge_model: str | None,
    source_results: Path | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    summary_path = output_path.with_suffix(".summary.json")
    summary = summarize(results)
    summary.update(
        {
            "dataset": str(dataset_path),
            "generated_at": datetime.now().astimezone().isoformat(),
            "mode": (
                "rejudge-existing-results"
                if source_results
                else "retrieval-only" if retrieval_only else "full-rag"
            ),
            "local_judge_model": judge_model,
        }
    )
    if source_results:
        summary["source_results"] = str(source_results)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)
        summary_file.write("\n")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV dataset (default: ~/Downloads/qa_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Per-question CSV output path (defaults to a timestamped file)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N dataset rows",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate retrieval without calling the Groq answer model",
    )
    parser.add_argument(
        "--judge-model",
        help="Optional local Ollama model for correctness, relevance, and grounding scores",
    )
    parser.add_argument(
        "--judge-results",
        type=Path,
        help="Rejudge an existing evaluation CSV without rerunning RAG generation",
    )
    parser.add_argument(
        "--trace-langsmith",
        action="store_true",
        help="Send LangChain run traces, prompts, retrieved context, and outputs to LangSmith",
    )
    parser.add_argument(
        "--langsmith-project",
        help="LangSmith project name (default: LANGSMITH_PROJECT or medical-chatbot-rag-eval)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.retrieval_only and args.judge_model:
        parser.error("--judge-model requires full-RAG mode")
    if args.judge_results and not args.judge_model:
        parser.error("--judge-results requires --judge-model")
    if args.judge_results and (args.retrieval_only or args.limit is not None):
        parser.error("--judge-results cannot be combined with --retrieval-only or --limit")

    os.chdir(PROJECT_ROOT)
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    try:
        configure_langsmith_tracing(
            enabled=args.trace_langsmith,
            project=args.langsmith_project,
        )
    except RuntimeError as error:
        parser.error(str(error))
    if args.trace_langsmith:
        print(
            "LangSmith tracing enabled; inputs, retrieved context, and outputs "
            "will be sent to the configured LangSmith project."
        )

    judge = build_local_judge(args.judge_model) if args.judge_model else None
    dataset_path = args.dataset.expanduser().resolve()
    source_results = None
    if args.judge_results:
        source_results = args.judge_results.expanduser().resolve()
        results = rejudge_existing_results(source_results, judge)
    else:
        rows = load_dataset(dataset_path)
        if args.limit is not None:
            rows = rows[: args.limit]
        runner = build_rag_runner(args.retrieval_only)
        results = evaluate_rows(rows, runner, judge=judge)

    if args.output:
        output_path = args.output
    elif source_results:
        output_path = source_results.with_name(
            f"{source_results.stem}_rejudged.csv"
        )
    else:
        output_path = (
            PROJECT_ROOT
            / "evaluation_results"
            / f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path = output_path.resolve()

    summary_path = write_results(
        results,
        output_path,
        dataset_path,
        args.retrieval_only,
        args.judge_model,
        source_results=source_results,
    )
    print(f"Evaluated {len(results)} questions.")
    print(f"Per-question results: {output_path}")
    print(f"Aggregate summary: {summary_path}")
    summary = summarize(results)
    print(
        f"Generation errors: {summary['generation_error_count']}/{summary['examples']}"
    )
    print(f"Judge errors: {summary['judge_error_count']}/{summary['examples']}")
    for metric_name, value in summary.items():
        if metric_name not in {
            "examples",
            "generation_error_count",
            "generation_error_ids",
            "judge_error_count",
            "judge_error_ids",
        }:
            formatted = "n/a" if value is None else f"{value:.1%}"
            print(f"{metric_name}: {formatted}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
