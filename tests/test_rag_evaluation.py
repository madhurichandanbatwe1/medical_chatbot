import json
import os
from pathlib import Path
import tempfile
import unittest
import csv
from unittest.mock import patch

from langchain_core.documents import Document

from scripts.evaluate_rag import (
    RESULT_COLUMNS,
    build_local_judge,
    configure_langsmith_tracing,
    entry_metadata_scores,
    evaluate_rows,
    key_fact_recall,
    rejudge_existing_results,
    reference_token_scores,
    score_with_local_judge,
)


class RagEvaluationTests(unittest.TestCase):
    def test_key_fact_recall_matches_normalized_phrases(self):
        score = key_fact_recall(
            ["beta blockers portal hypertension", "interferon hepatitis B and C"],
            "Beta blockers are used for portal hypertension. Interferon treats hepatitis B and C.",
        )

        self.assertEqual(score, 1.0)

    def test_entry_metrics_report_missing_entry_metadata(self):
        coverage, match_rate = entry_metadata_scores(
            [Document(page_content="cirrhosis information", metadata={"page": 4})],
            "Cirrhosis",
        )

        self.assertEqual(coverage, 0.0)
        self.assertIsNone(match_rate)

    def test_evaluate_rows_reports_retrieval_and_answer_coverage(self):
        rows = [
            {
                "id": "Q22",
                "entry": "Cirrhosis",
                "question": "What treatments are used?",
                "reference_answer": "Avoid alcohol; use beta blockers.",
                "key_facts": "avoid alcohol|beta blockers",
                "source": "medical reference",
            }
        ]

        def run_query(question):
            self.assertEqual(question, rows[0]["question"])
            return {
                "answer": "Avoid alcohol and consider beta blockers.",
                "context": [
                    Document(
                        page_content="Cirrhosis treatment includes avoid alcohol.",
                        metadata={"source": "medical.pdf", "page": 4},
                    )
                ],
            }

        result = evaluate_rows(rows, run_query)[0]

        self.assertEqual(result["context_key_fact_recall"], 0.5)
        self.assertEqual(result["answer_key_fact_recall"], 1.0)
        self.assertEqual(result["retrieved_entry_metadata_coverage"], 0.0)
        self.assertIsNone(result["retrieved_entry_match_rate"])
        self.assertGreater(result["reference_token_f1"], 0.0)

    def test_evaluate_rows_records_empty_generation_without_losing_retrieval(self):
        row = {
            "id": "Q01",
            "entry": "Cholestasis",
            "question": "What is cholestasis?",
            "reference_answer": "Bile excretion is interrupted.",
            "key_facts": "interruption in the excretion of bile",
            "source": "medical reference",
        }

        result = evaluate_rows(
            [row],
            lambda _: {
                "answer": None,
                "context": [Document(page_content="interruption in bile excretion")],
                "generation_error": "The RAG chain returned an empty answer",
            },
        )[0]

        self.assertEqual(result["context_key_fact_recall"], 1.0)
        self.assertIsNone(result["answer_key_fact_recall"])
        self.assertEqual(
            result["generation_error"],
            "The RAG chain returned an empty answer",
        )

    def test_evaluate_rows_records_bad_judge_json(self):
        row = {
            "id": "Q01",
            "entry": "Cholestasis",
            "question": "What is cholestasis?",
            "reference_answer": "Bile excretion is interrupted.",
            "key_facts": "",
            "source": "medical reference",
        }

        class InvalidJudge:
            def invoke(self, prompt):
                return type("JudgeResponse", (), {"content": "{invalid"})()

        result = evaluate_rows(
            [row],
            lambda _: {
                "answer": "Bile excretion is interrupted.",
                "context": [Document(page_content="Bile excretion is interrupted.")],
            },
            judge=InvalidJudge(),
        )[0]

        self.assertIn("invalid JSON", result["judge_error"])
        self.assertIsNone(result["judge_correctness"])

    def test_reference_token_scores_handles_empty_answer(self):
        self.assertEqual(
            reference_token_scores("reference answer", ""),
            (0.0, 0.0, 0.0),
        )

    def test_local_judge_receives_entry_and_returns_valid_scores(self):
        class StubJudge:
            def invoke(self, prompt):
                self.prompt = prompt
                return type(
                    "JudgeResponse",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "correctness": 1,
                                "answer_relevance": 1,
                                "faithfulness": 0.5,
                                "context_relevance": 1,
                                "notes": "One claim needs more evidence.",
                            }
                        )
                    },
                )()

        judge = StubJudge()
        scores = score_with_local_judge(
            judge=judge,
            question="What treatments are used?",
            expected_entry="Cirrhosis",
            reference_answer="Avoid alcohol.",
            key_facts=["avoid alcohol"],
            answer="Avoid alcohol.",
            context="Avoiding alcohol is part of treatment.",
        )

        self.assertIn("Expected entry: Cirrhosis", judge.prompt)
        self.assertEqual(scores["faithfulness"], 0.5)

    def test_local_judge_disables_qwen_reasoning_for_json_output(self):
        from unittest.mock import patch

        with patch("langchain_ollama.ChatOllama") as chat_ollama:
            build_local_judge("qwen3:8b")

        self.assertEqual(
            chat_ollama.call_args.kwargs["reasoning"],
            False,
        )

    def test_existing_results_can_be_rejudged_without_rag_generation(self):
        source_row = {
            "id": "Q01",
            "entry": "Cholestasis",
            "question": "What is cholestasis?",
            "reference_answer": "Bile excretion is interrupted.",
            "key_facts": "interruption in the excretion of bile",
            "answer": "Cholestasis is interrupted bile excretion.",
            "context_key_fact_recall": "1.0",
            "answer_key_fact_recall": "1.0",
            "reference_token_precision": "0.8",
            "reference_token_recall": "0.8",
            "reference_token_f1": "0.8",
            "retrieved_entry_metadata_coverage": "0.0",
            "retrieved_entry_match_rate": "",
            "retrieved_sources": "[]",
            "retrieved_contexts": json.dumps(
                [
                    {
                        "metadata": {"page": 10},
                        "page_content": "Bile excretion is interrupted.",
                    }
                ]
            ),
            "generation_error": "",
            "judge_correctness": "1.0",
            "judge_answer_relevance": "1.0",
            "judge_faithfulness": "1.0",
            "judge_context_relevance": "1.0",
            "judge_notes": "Old note",
            "judge_error": "",
        }

        class StubJudge:
            def invoke(self, prompt):
                self.prompt = prompt
                return type(
                    "JudgeResponse",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "correctness": 0.75,
                                "answer_relevance": 1.0,
                                "faithfulness": 0.5,
                                "context_relevance": 0.75,
                                "notes": "Evidence is only partially supported.",
                            }
                        )
                    },
                )()

        judge = StubJudge()
        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "previous.csv"
            with results_path.open("w", encoding="utf-8", newline="") as result_file:
                writer = csv.DictWriter(result_file, fieldnames=RESULT_COLUMNS)
                writer.writeheader()
                writer.writerow(source_row)

            results = rejudge_existing_results(results_path, judge)

        self.assertIn("[CHUNK 1; METADATA=", judge.prompt)
        self.assertEqual(results[0]["judge_correctness"], 0.75)
        self.assertEqual(
            results[0]["judge_notes"],
            "Evidence is only partially supported.",
        )

    def test_langsmith_tracing_requires_a_key_and_sets_project(self):
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}, clear=True):
            configure_langsmith_tracing(True, "medical-rag-test")

            self.assertEqual(os.environ["LANGSMITH_TRACING"], "true")
            self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "true")
            self.assertEqual(os.environ["LANGSMITH_PROJECT"], "medical-rag-test")

    def test_langsmith_tracing_is_off_by_default(self):
        with patch.dict(
            os.environ,
            {
                "LANGSMITH_TRACING": "true",
                "LANGCHAIN_TRACING_V2": "true",
                "LANGSMITH_API_KEY": "test-key",
            },
            clear=True,
        ):
            configure_langsmith_tracing(False)

            self.assertEqual(os.environ["LANGSMITH_TRACING"], "false")
            self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "false")

    def test_langsmith_tracing_without_a_key_is_an_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires LANGSMITH_API_KEY"):
                configure_langsmith_tracing(True)


if __name__ == "__main__":
    unittest.main()
