"""Provide synthetic local fixtures when private ticket CSVs are absent."""

from pathlib import Path

from generate_demo_data import DEFAULT_OUTPUT, generate

if not (DEFAULT_OUTPUT / "tickets.csv").exists() or not (DEFAULT_OUTPUT / "production_style_tickets.csv").exists() or not (DEFAULT_OUTPUT / "real_style_tickets_v2.csv").exists():
    generate(Path(DEFAULT_OUTPUT))
