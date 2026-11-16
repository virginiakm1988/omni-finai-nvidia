import tempfile
import unittest
from pathlib import Path
from project_demo.common import load_config, read_jsonl
from project_demo.data import example, prepare
from project_demo.inference import score_one, evaluate
from project_demo.training import recipe


class FinanceTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/default.json")

    def test_splits_disjoint(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            prepare(self.config, p)
            a, b, c = [
                {r["company"] for r in read_jsonl(p / (s + ".jsonl"))}
                for s in ["train", "validation", "test"]
            ]
            self.assertFalse(a & b or a & c or b & c)

    def test_reproducible(self):
        self.assertEqual(example(5, "test", 42), example(5, "test", 42))

    def test_reference_answers(self):
        for i in range(30):
            r = example(i, "test", 42)
            self.assertTrue(score_one({**r, "prediction": r["answer"]}))

    def test_wrong_extraction(self):
        r = example(0, "test", 42)
        self.assertFalse(
            score_one(
                {
                    **r,
                    "prediction": '{"revenue_million": 999999, "net_income_million":0}',
                }
            )
        )

    def test_reject_bool_numbers(self):
        self.assertFalse(
            score_one(
                {
                    "task": "extract",
                    "prediction": '{"revenue_million":true}',
                    "checks": {"revenue_million": 1},
                }
            )
        )

    def test_recipe_uses_correct_checkpoint(self):
        r = recipe(self.config, "original", Path("data"), Path("checkpoints"), 4)
        self.assertEqual(
            r["model"]["pretrained_model_name_or_path"],
            "ichanchiu/Llama-3.1-Omni-FinAI-70B",
        )
        self.assertEqual(r["seed"], 42)
        self.assertEqual(r["peft"]["dim"], 8)
        self.assertNotEqual(
            r["dataset"]["path_or_dataset_id"],
            r["validation_dataset"]["path_or_dataset_id"],
        )

    def test_empty_evaluation(self):
        with self.assertRaises(ValueError):
            evaluate([])
