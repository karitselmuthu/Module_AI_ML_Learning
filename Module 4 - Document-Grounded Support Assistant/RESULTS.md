# Module 4 local run results

These results come from the included fictional documents and hand-written questions. They check the workflow, not real support accuracy. Generated indexes, model weights, and adapters stay in ignored `artifacts/` and can be recreated with the commands in [README.md](README.md).

## Retrieval and grounded answers

| Run | Questions | Retrieval recall@3 | Mean reciprocal rank | Expected phrase | Grounded answer | Unsupported-question abstention |
|---|---:|---:|---:|---:|---:|---:|
| HashingVectorizer + FAISS | 9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| MiniLM + FAISS | 9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Seven questions have a source and two are intentionally unsupported. Retrieval recall and reciprocal rank use the seven answerable questions; abstention uses the two unsupported questions. The expected phrase and grounding rates also use the seven answerable questions. These small, source-specific checks do not distinguish the embedding methods reliably. A larger blind evaluation with paraphrases, distractor documents, and altered policies is needed.

The FLAN-T5 small prompt returned `five business days` for the approved-refund question. That phrase appears verbatim in `billing_policy.md`, so the application marked the answer `flan_verified` and attached that source. Other generated candidates are accepted only when they occur verbatim in retrieved evidence.

## LoRA routing exercise

The default eight-step run trained 172,032 adapter parameters out of 77,133,184 total parameters (0.223%). On six held-out synthetic tickets, the base model produced **0/6 valid correct labels** and the adapted model **2/6**. Several outputs were partial ticket text rather than one of the three required labels. The losses fluctuated across the eight single-example steps, from 6.06 at the first step to 3.01 at the last step, so this is not evidence of stable convergence. Longer training, more varied examples, and a larger validation set would be needed before claiming routing improvement.

The adapter exercise targets a stable output format. The document assistant's support facts remain in its FAISS index; rerun ingestion when those documents change.

## Checks performed

`python -m pytest -q`: **5 passed**. A disposable `warranty.txt` was indexed separately and answered with a `warranty.txt` citation. The included unknown-ticket workflow returned insufficient evidence.
