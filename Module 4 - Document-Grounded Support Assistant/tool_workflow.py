"""Week 10: an allowlisted, read-only support tool workflow."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from knowledge_assistant import ABSTAIN, DEFAULT_INDEX, ROOT, answer

STATUS_FILE = ROOT / "sample_data" / "ticket_status.json"
TICKET_ID = re.compile(r"\bDEMO-\d{4}\b", re.IGNORECASE)
STATUS_TERMS = {"status", "state", "progress", "update"}
ALLOWED_TOOLS = {"search_docs", "lookup_demo_ticket"}


def plan(question: str) -> dict:
    if not question.strip():
        raise ValueError("question must be nonempty")
    match = TICKET_ID.search(question)
    if match and any(term in question.lower().split() for term in STATUS_TERMS):
        return {"tool": "lookup_demo_ticket", "args": {"ticket_id": match.group().upper()}}
    return {"tool": "search_docs", "args": {"question": question}}


def execute(action: dict, index: Path = DEFAULT_INDEX, status_file: Path = STATUS_FILE) -> dict:
    tool = action.get("tool")
    if tool not in ALLOWED_TOOLS:
        raise ValueError("Tool is not on the allowlist")
    if tool == "search_docs":
        result = answer(action["args"]["question"], index)
    else:
        ticket_id = action["args"]["ticket_id"]
        if not TICKET_ID.fullmatch(ticket_id):
            raise ValueError("Only synthetic DEMO ticket IDs are supported")
        statuses = json.loads(Path(status_file).read_text())
        if ticket_id in statuses:
            result = {"status": "answered", "answer": f"{ticket_id}: {statuses[ticket_id]}",
                      "citations": [{"source": "sample_data/ticket_status.json", "ticket_id": ticket_id}]}
        else:
            result = {"status": "insufficient_evidence", "answer": ABSTAIN, "citations": []}
    return {"tool_call": action, "result": result}


def run(question: str, index: Path = DEFAULT_INDEX, status_file: Path = STATUS_FILE) -> dict:
    workflow = RunnableLambda(plan) | RunnableLambda(lambda action: execute(action, index, status_file))
    return workflow.invoke(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--status-file", type=Path, default=STATUS_FILE)
    args = parser.parse_args()
    print(json.dumps(run(args.question, args.index, args.status_file), indent=2))
