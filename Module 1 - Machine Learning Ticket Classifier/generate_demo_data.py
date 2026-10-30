"""Generate clearly synthetic ticket CSVs for a fresh public checkout."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "sample_data"
LABELS = ("account", "billing", "technical")
TOPICS = {
    "account": ("sign in", "password reset", "profile access", "team invitation", "account settings"),
    "billing": ("invoice", "payment", "subscription charge", "receipt", "tax amount"),
    "technical": ("server error", "file upload", "API timeout", "sync failure", "application crash"),
}


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict], force: bool) -> None:
    if path.exists() and not force:
        print(f"Keeping existing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {path} ({len(rows)} synthetic rows)")


def generate(output: Path = DEFAULT_OUTPUT, force: bool = False) -> None:
    short = []
    production = []
    v2 = []
    for label_index, label in enumerate(LABELS):
        for index in range(40):
            topic = TOPICS[label][index % 5]
            short.append({"text": f"Please help with my {topic}; example request {index + 1}.", "label": label})
        for scenario in range(20):
            topic = TOPICS[label][scenario % 5]
            for variant in range(10):
                number = label_index * 200 + scenario * 10 + variant + 1
                production.append({
                    "ticket_id": f"DEMO-P-{number:04d}",
                    "subject": f"Help with {topic}",
                    "body": f"Synthetic {label} request about {topic}; scenario {scenario + 1}, variation {variant + 1}.",
                    "label": label,
                    "scenario_id": f"DEMO-{label}-{scenario:02d}",
                })
        for index in range(200):
            topic = TOPICS[label][index % 5]
            route = LABELS[(label_index + 1) % 3] if index % 11 == 0 else label
            v2.append({
                "text": f"Synthetic {label} example about {topic}, item {index + 1}.",
                "label": label,
                "agent_routed_queue": route,
            })
    # Demonstrate unsupported labels and contradictory recorded routes.
    v2.append({"text": "Synthetic request outside the three supported queues.",
               "label": "other", "agent_routed_queue": "account"})
    v2.extend([
        {"text": "Synthetic billing route conflict example.", "label": "billing", "agent_routed_queue": "billing"},
        {"text": "Synthetic billing route conflict example.", "label": "billing", "agent_routed_queue": "technical"},
    ])
    write_csv(output / "tickets.csv", ("text", "label"), short, force)
    write_csv(output / "production_style_tickets.csv",
              ("ticket_id", "subject", "body", "label", "scenario_id"), production, force)
    write_csv(output / "real_style_tickets_v2.csv",
              ("text", "label", "agent_routed_queue"), v2, force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Replace existing CSV files in the output folder")
    args = parser.parse_args()
    generate(args.output, args.force)
