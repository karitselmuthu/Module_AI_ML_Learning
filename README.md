# Applied Machine Learning, Generative AI, and LLM Engineering

This repository contains hands-on projects for the first four modules of a [16-week learning program](16-Week-AI-Learning-Program.md). The program document is the full curriculum: weekly topics, practical work, milestones, assessment, and the roadmap through production deployment. This README is the starting point for running the projects already built.

## Projects available now

| Module | Weeks | Project | Start here |
|---|---|---|---|
| 1: Machine learning | 1–3 | Classify support tickets into account, billing, or technical queues | [Module 1 guide](<Module 1 - Machine Learning Ticket Classifier/README.md>) |
| 2: Deep learning | 4–6 | Categorize ticket attachments and compare a small attention model with a text baseline | [Module 2 guide](<Module 2 - Deep Learning Ticket Attachments/README.md>) |
| 3: Generative AI | 7–8 | Reconstruct images with a VAE and generate visual assets with saved prompts and review decisions | [Module 3 guide](<Module 3 - Generative AI Visual Assets/README.md>) |
| 4: LLMs and RAG | 9–10 | Answer from local support documents with citations, test retrieval and grounding, and compare a small LoRA adaptation | [Module 4 guide](<Module 4 - Document-Grounded Support Assistant/README.md>) |

Modules 5–7, covering multimodal applications, integration, and production workflows, are planned in the [full program](16-Week-AI-Learning-Program.md).

## Try a project

Each module has its own `requirements.txt`, instructions, tests, and results. Use Python 3.10 or newer. For the ticket classifier:

```bash
cd 'Module 1 - Machine Learning Ticket Classifier'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python ticket_classifier.py train --data sample_data/tickets.csv --output artifacts/sample_tickets --overwrite
python ticket_classifier.py predict 'My invoice has the wrong tax amount' --model artifacts/sample_tickets/model.joblib
python -m pytest -q
```

Start [Module 2](<Module 2 - Deep Learning Ticket Attachments/README.md>), [Module 3](<Module 3 - Generative AI Visual Assets/README.md>), and [Module 4](<Module 4 - Document-Grounded Support Assistant/README.md>) from their own folders using their guides. Their pretrained models may download weights on first use.

## Included learning data

- [Module 1 sample CSVs](<Module 1 - Machine Learning Ticket Classifier/sample_data/>) contain 120 short tickets, 600 scenario-based tickets, and 603 examples for comparing issue labels with routing labels.
- [Module 2 attachment images](<Module 2 - Deep Learning Ticket Attachments/data/generated/>) contain 240 synthetic screenshots and a train, validation, and test manifest. Module 3 uses these images for its VAE exercise.
- [Module 4 support guides](<Module 4 - Document-Grounded Support Assistant/sample_docs/>) and [evaluation questions](<Module 4 - Document-Grounded Support Assistant/eval/>) provide fictional policies, grounded answer checks, and ticket-routing examples.

All included examples are synthetic teaching data. Local ticket CSVs, trained model weights, and diffusion outputs are excluded from Git. The historical scores in module result reports came from earlier local experiments and may differ when you train on the included samples.

For the complete week-by-week syllabus and assessment plan, read [16-Week-AI-Learning-Program.md](16-Week-AI-Learning-Program.md).
