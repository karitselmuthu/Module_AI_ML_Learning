# Module 1 project result

Run date: 26 October 2026. Dataset: 120 hand-written examples, evenly divided across account, billing, and technical queues. Random seed: 42. Full metrics, parameters, and confusion matrix are in `artifacts/metrics.json`; individual held-out predictions are in `artifacts/test_predictions.csv`. These existing test results have been reviewed during model development and are exploratory, not a fresh final evaluation. The public repository includes different synthetic CSVs in `sample_data/`, so rerunning with them will not reproduce these numbers.

| Evaluation | Majority baseline macro F1 | TF-IDF + logistic regression macro F1 |
|---|---:|---:|
| Validation (24 tickets) | 0.1667 | 0.9582 |
| Held-out test (24 tickets) | 0.1667 | 0.7111 |

Three-fold cross-validation on the 72 training examples gave macro F1 **0.8039** for the selected configuration. After validation, the model was refitted on 96 training and validation examples before its one held-out test evaluation. The test score is lower than the validation score, which is a reason to gather more representative data rather than treat the validation result as expected production performance.

## Error analysis

The final model got 17 of 24 held-out tickets correct. Account recall was 5/8, billing recall 6/8, and technical recall 6/8.

- A billing-address request went to account. Profile and address wording likely competed with the billing cue.
- Two team-member access requests went to billing. Seat and subscription changes can overlap with account administration.
- An integration-sync request went to billing, and an attachment-opening problem went to account. These technical cases are not well represented in 40 examples.
- A missing account-confirmation message went to billing. A short text with a weak category signal is hard to route.
- A billed-total dispute went to technical. It referred to a page but described a billing issue.

These are observations from seven specific misses, not proof of their causes. The next data collection should include cross-category language and real tickets, with labels reviewed by support staff.

## Production-style data experiment

The added `data/production_style_tickets.csv` has 600 rows. After removing seven exact duplicate combined subject/body texts, 593 remain. Scenario-grouped splitting produced 355 training, 119 validation, and 119 test examples. No `scenario_id` appears in more than one split; the same grouping is used in cross-validation.

| Model on the same 119 held-out production-style tickets | Macro F1 |
|---|---:|
| Majority-class baseline | 0.2159 |
| Original 120-ticket model | 0.6772 |
| Model trained on production-style tickets | 0.6173 |

The production-style model's training cross-validation macro F1 was 0.6944 and validation macro F1 was 0.7950. Its test recall was 11/35 for account, 16/27 for billing, and 53/57 for technical. Many account tickets were routed to technical (23/35). Adding more examples did not improve this scenario-held-out score. The original model remains the default CLI artifact for continuity; the new one is saved separately under `artifacts/production_style_tickets/`. These are small, previously inspected test sets, and further model comparisons require fresh evaluation data.

## Real-style v2 target choice

One local ticket has issue `label=billing`, while `agent_routed_queue=technical`. The saved issue-label model predicts billing for that ticket, consistent with its training target. The queue-target model, trained separately with `--target agent_routed_queue`, also predicts billing for that individual ticket. One ticket's recorded route cannot force a fitted model to assign that route to similar text.

The older queue-target model's scenario-held-out test macro F1 was **0.6825**, versus **0.2109** for its majority-class baseline. This evaluates imitation of recorded routing, a different task from classifying the issue type. Its artifacts remain in `artifacts/real_style_tickets_v2_agent_routed_queue/`.

### Reviewed v2 data policy

The v2 audit found 1,500 source rows, 359 disagreements between `label` and `agent_routed_queue`, and 194 rows with issue label `other`. The issue-label run explicitly excluded those unsupported rows and 28 same-label duplicates, retaining **1,278** examples. The queue-target audit found 15 identical-text groups with different routed queues (36 rows). The reviewed queue run excluded all 36 conflicting rows plus 16 remaining same-queue duplicates, retaining **1,448** examples. No contradictory duplicate was resolved by choosing the first row.

| Reviewed v2 run | Training CV macro F1 | Validation macro F1 | Existing scenario-held-out test macro F1 |
|---|---:|---:|---:|
| Issue label, `artifacts/real_style_tickets_v2_reviewed/` | 0.6473 | 0.7131 | 0.9215 |
| Routed queue, `artifacts/real_style_tickets_v2_queue_reviewed/` | 0.5700 | 0.6428 | 0.6937 |

The large difference between issue-label validation and test scores is another reason not to treat this single split as a stable performance estimate. The reviewed models include a target-specific prediction field and source metadata. Use future tickets with reviewed labels, collected after model selection, for a final evaluation; do not select a winner from the previously inspected test scores.

## Learning review

- **Covered by this project:** problem framing, pandas/NumPy/Matplotlib exploration, classification, text feature engineering, split design, pipelines, baseline comparison, cross-validation, tuning, macro F1, confusion matrix, and error review.
- **Follow-up topics from the module outline:** practice regression on a separate numeric target and clustering on an unlabeled dataset. Compare with the learner's detailed Module 1 breakdown when it is available.
