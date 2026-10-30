import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ticket_classifier import (
    DATA, LABELS, PRODUCTION_STYLE_DATA, evaluate_external, load_data, predict, split_data, train,
)


class WorkflowTests(unittest.TestCase):
    def test_splits_are_disjoint_and_stratified(self):
        frame = load_data(DATA)
        train_frame, valid_frame, test_frame = split_data(frame)
        sets = [set(part["text"]) for part in (train_frame, valid_frame, test_frame)]
        self.assertEqual(sum(map(len, sets)), len(frame))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for part in (train_frame, valid_frame, test_frame):
            self.assertEqual(set(part["label"]), set(LABELS))

    def test_training_produces_comparable_reports(self):
        with TemporaryDirectory() as directory:
            result = train(DATA, Path(directory))
            self.assertTrue((Path(directory) / "model.joblib").exists())
            self.assertTrue((Path(directory) / "model_metadata.json").exists())
            self.assertEqual(result["split_rows"], {"train": 72, "validation": 24, "test": 24})
            self.assertGreater(result["test"]["model"]["macro_f1"], result["test"]["baseline"]["macro_f1"])
            self.assertEqual(predict(Path(directory) / "model.joblib", "Please send my invoice")["target_column"], "label")
            with self.assertRaisesRegex(ValueError, "identical"):
                evaluate_external(Path(directory) / "model.joblib", DATA)
            external = Path(directory) / "new_tickets.csv"
            pd.DataFrame([
                {"text": f"Fresh external example {label} number {index}", "label": label}
                for label in LABELS for index in range(5)
            ]).to_csv(external, index=False)
            evaluation = evaluate_external(Path(directory) / "model.joblib", external)
            self.assertEqual(evaluation["data_audit"]["retained_rows"], 15)
            copied = pd.read_csv(external)
            copied.loc[0, "text"] = load_data(DATA)["text"].iloc[0]
            copied.to_csv(external, index=False)
            with self.assertRaisesRegex(ValueError, "ticket texts from the training source"):
                evaluate_external(Path(directory) / "model.joblib", external)
            with self.assertRaises(FileExistsError):
                train(DATA, Path(directory))
            train(DATA, Path(directory), overwrite=True)

    def test_production_style_scenarios_do_not_cross_splits(self):
        frame = load_data(PRODUCTION_STYLE_DATA)
        self.assertGreater(len(frame), 500)
        self.assertTrue(frame["text"].is_unique)
        train_frame, valid_frame, test_frame = split_data(frame)
        scenario_sets = [set(part["scenario_id"]) for part in (train_frame, valid_frame, test_frame)]
        self.assertFalse(scenario_sets[0] & scenario_sets[1])
        self.assertFalse(scenario_sets[0] & scenario_sets[2])
        self.assertFalse(scenario_sets[1] & scenario_sets[2])
        self.assertEqual(set(frame["label"]), set(LABELS))

    def test_alternate_target_uses_routed_queue(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tickets.csv"
            rows = [
                {"text": f"Ticket {label} {index}", "label": "billing", "agent_routed_queue": label}
                for label in LABELS for index in range(5)
            ]
            pd.DataFrame(rows).to_csv(path, index=False)
            frame = load_data(path, target_column="agent_routed_queue")
            self.assertEqual(frame["label"].value_counts().to_dict(), {label: 5 for label in LABELS})

    def test_v2_requires_explicit_data_exclusions(self):
        path = DATA.with_name("real_style_tickets_v2.csv")
        with self.assertRaisesRegex(ValueError, "unsupported target rows"):
            load_data(path)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            load_data(path, target_column="agent_routed_queue")
        issue_frame, issue_audit = load_data(
            path, drop_unsupported=True, return_audit=True
        )
        queue_frame, queue_audit = load_data(
            path, target_column="agent_routed_queue",
            drop_conflicts=True, return_audit=True,
        )
        self.assertGreater(issue_audit["unsupported_target_counts"].get("other", 0), 0)
        self.assertGreater(queue_audit["conflicting_text_groups"], 0)
        self.assertGreater(queue_audit["issue_vs_route_disagreements"], 0)
        self.assertEqual(
            queue_audit["source_rows"] - queue_audit["blank_text_rows"]
            - queue_audit["unsupported_target_rows"] - queue_audit["conflicting_text_rows"]
            - queue_audit["same_target_duplicate_rows"],
            queue_audit["retained_rows"],
        )
        self.assertTrue(issue_frame["text"].is_unique)
        self.assertTrue(queue_frame["text"].is_unique)

    def test_queue_model_records_target_and_exclusions(self):
        path = DATA.with_name("real_style_tickets_v2.csv")
        with TemporaryDirectory() as directory:
            result = train(
                path, Path(directory), target_column="agent_routed_queue",
                drop_conflicts=True,
            )
            self.assertGreater(result["data_audit"]["conflicting_text_rows"], 0)
            prediction = predict(Path(directory) / "model.joblib", "The payment failed")
            self.assertEqual(prediction["target_column"], "agent_routed_queue")
            self.assertIn("routed_queue", prediction)
            self.assertNotIn("issue_label", prediction)


if __name__ == "__main__":
    unittest.main()
