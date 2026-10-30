# Module 2 training report

Run date: 26 October 2026. Random seed: 42. All results are **exploratory** because the data is small or synthetic and the holdouts have been inspected during development.

## Image categorization

| Model | Training images | Validation images | Test images | Best validation macro F1 | Test macro F1 |
|---|---:|---:|---:|---:|---:|
| CNN from scratch | 144 | 48 | 48 | 1.0000 (epoch 23) | 1.0000 |
| Pretrained MobileNetV3 Small, frozen features | 144 | 48 | 48 | 1.0000 (epoch 1) | 1.0000 |

Both models classified all 48 generated test images correctly. This is expected for a small, highly structured dataset with category-specific screenshot elements. It does **not** show that either model would work on real invoices, photographs, blurred screenshots, unseen languages, or mixed-category documents. The MobileNet model reached the maximum validation score much earlier, illustrating the effect of pretrained features on this task; this single run does not prove it is generally superior.

The scratch CNN's training macro F1 rose from about 0.19 in the first epoch to 0.92 in the last, while validation macro F1 reached 1.00. The best checkpoint was at epoch 23. The transfer model reached 1.00 validation macro F1 after epoch 1 and remained high; its best validation checkpoint was retained even though training continued for the patience window. View `artifacts/scratch/training_curves.png` and `artifacts/mobilenet/training_curves.png` to inspect loss and F1 rather than relying only on final accuracy.

There are no image misclassifications to analyze in this generated test set. The next error-analysis step requires harder, independently sourced images. Useful categories include low-resolution screenshots, cropped invoices, login errors that mention payments, and technical alerts embedded in bills.

## Text self-attention comparison

| Model | Best validation macro F1 | Test macro F1 |
|---|---:|---:|
| Tiny transformer with token and position embeddings | 0.7260 (epoch 8) | 0.5270 |
| TF-IDF + logistic regression baseline | 0.9582 | 0.6711 |

The attention model reached 1.00 training macro F1 by the end of its run, while its validation macro F1 was about 0.71 and its validation loss remained much higher than training loss. That gap is evidence of **overfitting** on 72 training tickets. The baseline is the demonstration app's default because its validation score is higher. Test scores were also inspected during development, so this is not an untouched final model comparison. The models were trained on the same training partition and evaluated on the same validation and test partitions; neither result estimates real traffic performance.

The attention model missed 10 of 24 test tickets; the TF-IDF baseline missed 8. Both routed “How do I update the billing address” to account. The attention model also routed “Please split this payment across two cards” to account and “The password reset link has expired” to technical. These examples show where sparse data and overlapping language cause mistakes; they do not prove a single root cause. Inspect `artifacts/text_attention/test_predictions.csv` for all disagreements. More diverse labeled tickets, a pretrained language encoder, or stronger regularization would be reasonable experiments, with model choice made on validation data and a new final test set reserved.

## Assessment answers

- **Training curves:** The scratch CNN learned gradually; the pretrained CNN learned quickly on these images; the text attention model shows a training-validation gap.
- **Overfitting:** Strong training performance with weaker validation performance means the model learned patterns specific to the small training set. Dropout and early stopping limit it but do not make the data representative.
- **Model choice:** MobileNetV3 Small is useful for demonstrating transfer learning, while the scratch CNN exposes the full optimization loop. TF-IDF remains the default text model because its validation macro F1 was higher.
- **Next evidence needed:** An independently collected, labeled image set and future ticket texts, with customer or scenario separation and a label policy for ambiguous and unsupported cases.
