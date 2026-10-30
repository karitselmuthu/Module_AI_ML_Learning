"""Train and use a small support-ticket routing classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
LOCAL_DATA = ROOT / "data"
SAMPLE_DATA = ROOT / "sample_data"
DATA_DIR = (LOCAL_DATA if all((LOCAL_DATA / name).is_file() for name in
                              ("tickets.csv", "production_style_tickets.csv", "real_style_tickets_v2.csv"))
            else SAMPLE_DATA)
DATA = DATA_DIR / "tickets.csv"
PRODUCTION_STYLE_DATA = DATA_DIR / "production_style_tickets.csv"
ARTIFACTS = ROOT / "artifacts"
LABELS = ("account", "billing", "technical")
SEED = 42


def default_output_dir(data_path: Path, target_column: str = "label") -> Path:
    data_path = Path(data_path)
    if data_path.resolve() == DATA.resolve() and target_column == "label":
        return ARTIFACTS
    suffix = "_" + target_column if target_column != "label" else ""
    return ARTIFACTS / (data_path.stem + suffix)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_data(path: Path = DATA, target_column: str = "label", *,
              drop_unsupported: bool = False, drop_conflicts: bool = False,
              return_audit: bool = False):
    frame = pd.read_csv(path)
    source_rows = len(frame)
    issue_route_disagreements = None
    if {"label", "agent_routed_queue"}.issubset(frame.columns):
        issue_route_disagreements = int((
            frame["label"].fillna("").astype(str).str.strip().str.lower()
            != frame["agent_routed_queue"].fillna("").astype(str).str.strip().str.lower()
        ).sum())
    if target_column not in frame.columns:
        raise ValueError(f"CSV must contain target column: {target_column}")
    if target_column != "label":
        frame["label"] = frame[target_column]
    if {"subject", "body", "label"}.issubset(frame.columns):
        # Operational metadata, especially scenario_id, must never become a feature.
        frame["text"] = (
            frame["subject"].fillna("").astype(str).str.strip() + "\n\n"
            + frame["body"].fillna("").astype(str).str.strip()
        )
        columns = ["text", "label"] + [
            column for column in ("ticket_id", "scenario_id", "created_at") if column in frame
        ]
        frame = frame[columns].copy()
    elif {"text", "label"}.issubset(frame.columns):
        frame = frame[["text", "label"]].copy()
    else:
        raise ValueError("CSV must contain text/label or subject/body/label columns")
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["label"] = frame["label"].fillna("").astype(str).str.strip().str.lower()
    blank_rows = int(frame["text"].eq("").sum())
    frame = frame[frame["text"] != ""].copy()
    unsupported = frame[~frame["label"].isin(LABELS)]
    unsupported_counts = unsupported["label"].value_counts(dropna=False).to_dict()
    if len(unsupported) and not drop_unsupported:
        raise ValueError(
            f"Found {len(unsupported)} unsupported target rows {unsupported_counts}; "
            "review them or pass --drop-unsupported"
        )
    frame = frame[frame["label"].isin(LABELS)].copy()
    conflicting_texts = frame.groupby("text")["label"].nunique()
    conflicting_texts = set(conflicting_texts[conflicting_texts > 1].index)
    conflict_rows = frame[frame["text"].isin(conflicting_texts)]
    if len(conflict_rows) and not drop_conflicts:
        raise ValueError(
            f"Found {len(conflicting_texts)} identical-text groups with conflicting "
            f"{target_column} values ({len(conflict_rows)} rows); "
            "review them or pass --drop-conflicts"
        )
    frame = frame[~frame["text"].isin(conflicting_texts)].copy()
    duplicate_rows = int(frame.duplicated(subset=["text"]).sum())
    frame = frame.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    audit = {
        "source_rows": source_rows,
        "blank_text_rows": blank_rows,
        "unsupported_target_rows": len(unsupported),
        "unsupported_target_counts": unsupported_counts,
        "conflicting_text_groups": len(conflicting_texts),
        "conflicting_text_rows": len(conflict_rows),
        "conflicting_ticket_ids": (conflict_rows["ticket_id"].astype(str).tolist()
                                   if "ticket_id" in conflict_rows else []),
        "same_target_duplicate_rows": duplicate_rows,
        "retained_rows": len(frame),
        "target_column": target_column,
        "issue_vs_route_disagreements": issue_route_disagreements,
    }
    if "scenario_id" in frame and frame["scenario_id"].isna().any():
        raise ValueError("scenario_id must be populated for group-safe splitting")
    if not return_audit and (frame["label"].nunique() != len(LABELS)
                             or frame["label"].value_counts().min() < 5):
        raise ValueError("Need at least five valid, unique tickets per label")
    return (frame, audit) if return_audit else frame


def split_data(frame: pd.DataFrame):
    # The held-out test set is never seen by tuning or model selection.
    if "scenario_id" in frame:
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_valid_idx, test_idx = next(
            outer.split(frame["text"], frame["label"], groups=frame["scenario_id"])
        )
        train_valid, test = frame.iloc[train_valid_idx], frame.iloc[test_idx]
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
        train_idx, valid_idx = next(
            inner.split(train_valid["text"], train_valid["label"], groups=train_valid["scenario_id"])
        )
        train, valid = train_valid.iloc[train_idx], train_valid.iloc[valid_idx]
    else:
        train_valid, test = train_test_split(
            frame, test_size=0.20, random_state=SEED, stratify=frame["label"]
        )
        train, valid = train_test_split(
            train_valid, test_size=0.25, random_state=SEED, stratify=train_valid["label"]
        )
    return train, valid, test


def metrics(model, frame: pd.DataFrame) -> dict:
    truth = frame["label"].tolist()
    predicted = model.predict(frame["text"]).tolist()
    return {
        "macro_f1": round(f1_score(truth, predicted, average="macro"), 4),
        "classification_report": classification_report(
            truth, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
        "labels": list(LABELS),
    }


def train(data_path: Path = DATA, output_dir: Path | None = None,
          target_column: str = "label", *, drop_unsupported: bool = False,
          drop_conflicts: bool = False, overwrite: bool = False) -> dict:
    data_path = Path(data_path)
    output_dir = Path(output_dir) if output_dir is not None else default_output_dir(data_path, target_column)
    artifact_names = ("model.joblib", "model_metadata.json", "metrics.json", "test_predictions.csv")
    if not overwrite and any((output_dir / name).exists() for name in artifact_names):
        raise FileExistsError(f"Artifacts already exist in {output_dir}; choose --output or --overwrite")
    frame, audit = load_data(
        data_path, target_column=target_column,
        drop_unsupported=drop_unsupported, drop_conflicts=drop_conflicts,
        return_audit=True,
    )
    if frame["label"].nunique() != len(LABELS) or frame["label"].value_counts().min() < 5:
        raise ValueError("Need at least five valid, unique tickets per label")
    train_frame, valid_frame, test_frame = split_data(frame)
    # Fit all text processing within each CV fold, then compare against a majority baseline.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train_frame["text"], train_frame["label"])
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    grouped = "scenario_id" in train_frame
    cv = (StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
          if grouped else StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED))
    search = GridSearchCV(
        pipeline,
        {"tfidf__ngram_range": [(1, 1), (1, 2)], "classifier__C": [0.1, 1.0, 10.0]},
        scoring="f1_macro", cv=cv, n_jobs=1, refit=True,
    )
    search.fit(
        train_frame["text"], train_frame["label"],
        **({"groups": train_frame["scenario_id"]} if grouped else {}),
    )
    validation_baseline = metrics(baseline, valid_frame)
    validation_model = metrics(search.best_estimator_, valid_frame)

    # Refit the chosen configuration using train + validation only, then evaluate once on test.
    selected = search.best_estimator_.set_params(**search.best_params_)
    fit_frame = pd.concat([train_frame, valid_frame], ignore_index=True)
    selected.fit(fit_frame["text"], fit_frame["label"])
    baseline.fit(fit_frame["text"], fit_frame["label"])
    test_baseline = metrics(baseline, test_frame)
    test_model = metrics(selected, test_frame)
    prediction_columns = ["label"] + (["ticket_id"] if "ticket_id" in test_frame else ["text"])
    predictions = test_frame[prediction_columns].copy()
    predictions["predicted"] = selected.predict(test_frame["text"])
    predictions["correct"] = predictions["label"] == predictions["predicted"]
    result = {
        "data_file": str(data_path),
        "data_sha256": file_sha256(data_path),
        "target_column": target_column,
        "data_audit": audit,
        "evaluation_status": "exploratory_existing_dataset",
        "split_strategy": "stratified_group_by_scenario" if grouped else "stratified_random",
        "dataset_rows": len(frame),
        "split_rows": {"train": len(train_frame), "validation": len(valid_frame), "test": len(test_frame)},
        "label_counts": frame["label"].value_counts().sort_index().to_dict(),
        "selection_metric": "macro_f1",
        "cv_best_macro_f1": round(search.best_score_, 4),
        "best_params": search.best_params_,
        "validation": {"baseline": validation_baseline, "model": validation_model},
        "test": {"baseline": test_baseline, "model": test_model},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, output_dir / "model.joblib")
    metadata = {
        "target_column": target_column,
        "data_file": str(data_path),
        "data_sha256": result["data_sha256"],
        "source_text_sha256": sorted({text_sha256(text) for text in frame["text"]}),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": list(LABELS),
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    return result


def predict(model_path: Path, text: str) -> dict:
    model_path = Path(model_path)
    model = joblib.load(model_path)
    probabilities = model.predict_proba([text])[0]
    metadata_path = model_path.with_name("model_metadata.json")
    metrics_path = model_path.with_name("metrics.json")
    if metadata_path.exists():
        target_column = json.loads(metadata_path.read_text())["target_column"]
    elif metrics_path.exists():
        target_column = json.loads(metrics_path.read_text()).get("target_column", "label")
    else:
        target_column = "label"
    return {
        "target_column": target_column,
        "routed_queue" if target_column == "agent_routed_queue" else "issue_label": str(model.predict([text])[0]),
        "probabilities": {str(label): round(float(probability), 4)
                          for label, probability in zip(model.classes_, probabilities)},
    }


def evaluate_external(model_path: Path, data_path: Path, *,
                      drop_unsupported: bool = False, drop_conflicts: bool = False) -> dict:
    model_path, data_path = Path(model_path), Path(data_path)
    metadata_path = model_path.with_name("model_metadata.json")
    if not metadata_path.exists():
        raise ValueError("External evaluation requires a newly trained model with model_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    if file_sha256(data_path) == metadata["data_sha256"]:
        raise ValueError("Evaluation CSV is identical to the model's training source")
    frame, audit = load_data(
        data_path, target_column=metadata["target_column"],
        drop_unsupported=drop_unsupported, drop_conflicts=drop_conflicts,
        return_audit=True,
    )
    if frame["label"].nunique() != len(LABELS) or frame["label"].value_counts().min() < 5:
        raise ValueError("External evaluation needs at least five tickets per target class")
    known_hashes = set(metadata.get("source_text_sha256", []))
    if not known_hashes:
        raise ValueError("Model metadata lacks source text hashes; retrain before external evaluation")
    overlap = sum(text_sha256(text) in known_hashes for text in frame["text"])
    if overlap:
        raise ValueError(f"Evaluation CSV contains {overlap} ticket texts from the training source")
    model = joblib.load(model_path)
    return {
        "evaluation_status": "external_data_no_exact_text_overlap",
        "target_column": metadata["target_column"],
        "training_data_sha256": metadata["data_sha256"],
        "evaluation_data_sha256": file_sha256(data_path),
        "data_audit": audit,
        "metrics": metrics(model, frame),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="train and evaluate")
    train_parser.add_argument("--data", type=Path, default=DATA)
    train_parser.add_argument("--output", type=Path, default=None)
    train_parser.add_argument("--target", choices=("label", "agent_routed_queue"), default="label")
    train_parser.add_argument("--drop-unsupported", action="store_true")
    train_parser.add_argument("--drop-conflicts", action="store_true")
    train_parser.add_argument("--overwrite", action="store_true")
    audit_parser = subparsers.add_parser("audit", help="inspect exclusions and target disagreement")
    audit_parser.add_argument("--data", type=Path, default=DATA)
    audit_parser.add_argument("--target", choices=("label", "agent_routed_queue"), default="label")
    eval_parser = subparsers.add_parser("evaluate", help="score a saved model on new labeled data")
    eval_parser.add_argument("--data", type=Path, required=True)
    eval_parser.add_argument("--model", type=Path, required=True)
    eval_parser.add_argument("--drop-unsupported", action="store_true")
    eval_parser.add_argument("--drop-conflicts", action="store_true")
    predict_parser = subparsers.add_parser("predict", help="route a new ticket")
    predict_parser.add_argument("text")
    predict_parser.add_argument("--model", type=Path, default=ARTIFACTS / "model.joblib")
    args = parser.parse_args()
    try:
        if args.command == "audit":
            _, audit = load_data(
                args.data, target_column=args.target,
                drop_unsupported=True, drop_conflicts=True, return_audit=True,
            )
            print(json.dumps(audit, indent=2))
        elif args.command == "evaluate":
            print(json.dumps(evaluate_external(
                args.model, args.data,
                drop_unsupported=args.drop_unsupported,
                drop_conflicts=args.drop_conflicts,
            ), indent=2))
        elif args.command == "train":
            output_dir = args.output or default_output_dir(args.data, args.target)
            result = train(
                args.data, output_dir, target_column=args.target,
                drop_unsupported=args.drop_unsupported,
                drop_conflicts=args.drop_conflicts,
                overwrite=args.overwrite,
            )
            print(json.dumps({
                "target_column": result["target_column"],
                "data_audit": result["data_audit"],
                "cv_macro_f1": result["cv_best_macro_f1"],
                "validation_macro_f1": result["validation"]["model"]["macro_f1"],
                "test_macro_f1": result["test"]["model"]["macro_f1"],
                "baseline_test_macro_f1": result["test"]["baseline"]["macro_f1"],
                "artifacts": str(output_dir),
            }, indent=2))
        else:
            print(json.dumps(predict(args.model, args.text), indent=2))
    except (ValueError, FileExistsError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
