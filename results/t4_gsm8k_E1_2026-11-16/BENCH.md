# Live benchmark: gsm8k (300 questions)

Calibration: `q35-0.8b` free width 16 (R² 0.998, Tesla T4); `q35-4b` free width 4 (R² 0.997, Tesla T4)

| policy | accuracy | mean s | median s | p90 s | mean J | mean tokens | escalated | big model started | verified |
|---|---|---|---|---|---|---|---|---|---|
| overlap | 0.893 | 34.45 | 19.33 | 73.90 | 2848.5 | 7380 | 0.27 | 0.34 | 0.63 |
| big_sc2 | 0.957 | 43.43 | 40.23 | 62.04 | 5713.2 | 3200 | 0.00 | 0.00 | 0.00 |
| repair | 0.840 | 60.03 | 22.49 | 185.24 | 3967.7 | 12681 | 0.14 | 0.14 | 0.09 |

(Written by 0.5.0. Its "big model started" column was escalation-or-trigger based and reported 0 for big_sc2; `report/REPORT.md` has the corrected "largest model decoded" column.)
