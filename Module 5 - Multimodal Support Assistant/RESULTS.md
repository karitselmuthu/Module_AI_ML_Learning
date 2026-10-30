# Module 5 local sample results

The fictional sample media passed the fixed expected-phrase checks in one CPU run. This is a workflow check, not an independent accuracy estimate.

| Input | Phrase check | Processing time |
|---|---|---:|
| Text | Password and refunds found | <0.001 s (reported as 0.000) |
| Image | `BILLING RECEIPT` and `$42.00` found | 1.036 s |
| Video with supplied subtitles | Billing page, receipt download, and subtitle phrase found | 1.143 s |
| Audio | “billing receipt” found | 0.739 s |

`evaluate.py` reported **4/4** expected-phrase matches. Empty text, blank image, and one-second silent WAV returned `insufficient_input`. Model startup and media size change timing. The supplied `.srt` differs from the actual audio track; inspect the `kind: transcript` entries for each video run. The visual summary can omit events between sampled frames.

`python -m pytest -q`: **7 passed**, including extraction from the checked-in text PDF. A local macOS voice check produced a decodable 1.58-second WAV. Automated tests mock OCR and speech recognition where appropriate; the assessment script uses the real backends.
