# Live benchmark: gsm8k (300 questions)

Calibration: `q35-0.8b` free width 16 (R² 0.997, Tesla T4); `q35-4b` free width 4 (R² 0.998, Tesla T4)

| policy | n | accuracy | mean s | median s | p90 s | mean J | mean tokens | escalated | largest model decoded | verified |
|---|---|---|---|---|---|---|---|---|---|---|
| overlap | 300 | 0.937 | 37.23 | 20.94 | 74.65 | 3186.2 | 7677 | 0.35 | 0.43 | 0.64 |
