# Live benchmark: gsm8k (100 questions)

Calibration: `q35-0.8b` free width 16 (R² 0.998, Tesla T4); `q35-4b` free width 4 (R² 0.997, Tesla T4)

| policy | accuracy | mean s | p90 s | mean J | mean tokens | escalated | verified |
|---|---|---|---|---|---|---|---|
| sc | 0.680 | 38.54 | 46.38 | 2528.9 | 4255 | 0.00 | 0.00 |
| hwave | 0.790 | 48.53 | 139.89 | 3232.1 | 10025 | 0.18 | 0.00 |
| freewidth | 0.780 | 41.53 | 105.19 | 2712.5 | 9591 | 0.07 | 0.11 |
