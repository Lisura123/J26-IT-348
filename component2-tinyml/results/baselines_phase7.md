# Phase 7: baseline comparison (TRAINING window only)

Dataset SHA-256 `5fbb93471c056c274cbf81e8c317934e379b2484013147e4c2cacfa93421595b`. Python 3.13.16, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1. Seeds for stochastic models: [0, 1, 2].

The OPERATION window was not used. Reading-level metrics are computed on classified readings; coverage is reported separately. **Pooled** = one metric over the combined held-out predictions of all folds; **fold mean/min** = statistics over per-fold values. Recall is not reported for classes absent from a held-out partition. Size = learned parameters or tree nodes (an indicator only, not measured flash/RAM).

## cv

### Reading-level metrics

| Feature set | Model | pooled macro-F1 | fold mean | fold std | fold min | pooled OPEN recall | pooled SHORT recall | size | fits |
|---|---|---|---|---|---|---|---|---|---|
| physics7_no_v_deficit | forest | 0.991 | 0.992 | 0.019 | 0.933 | 1.000 | 1.000 | 278 | 15 |
| physics7_no_v_deficit | logreg | 0.980 | 0.980 | 0.045 | 0.899 | 1.000 | 1.000 | 40 | 5 |
| physics7_no_v_deficit | mlp | 0.980 | 0.979 | 0.044 | 0.893 | 1.000 | 1.000 | 213 | 15 |
| physics7_no_v_deficit | rules | 0.920 | 0.919 | 0.106 | 0.746 | 1.000 | 1.000 | 4 | 5 |
| physics7_no_v_deficit | tree | 0.911 | 0.899 | 0.154 | 0.634 | 1.000 | 1.000 | 9 | 5 |
| physics8 | logreg | 0.982 | 0.981 | 0.042 | 0.907 | 1.000 | 1.000 | 45 | 5 |
| physics8 | mlp | 0.981 | 0.981 | 0.040 | 0.895 | 1.000 | 1.000 | 229 | 15 |
| physics8 | forest | 0.975 | 0.973 | 0.043 | 0.874 | 1.000 | 1.000 | 264 | 15 |
| physics8 | rules | 0.920 | 0.919 | 0.106 | 0.746 | 1.000 | 1.000 | 4 | 5 |
| physics8 | tree | 0.896 | 0.881 | 0.100 | 0.723 | 1.000 | 1.000 | 9 | 5 |
| raw8 | mlp | 0.958 | 0.952 | 0.062 | 0.865 | 1.000 | 1.000 | 229 | 15 |
| raw8 | logreg | 0.957 | 0.951 | 0.067 | 0.865 | 1.000 | 1.000 | 45 | 5 |
| raw8 | tree | 0.914 | 0.911 | 0.088 | 0.800 | 1.000 | 1.000 | 9 | 5 |
| raw8 | forest | 0.902 | 0.902 | 0.093 | 0.781 | 1.000 | 0.967 | 454 | 15 |

### Run-level metrics (majority vote per run, pooled over folds)

| Feature set | Model | runs | majority-vote accuracy | worst single-run reading accuracy |
|---|---|---|---|---|
| physics7_no_v_deficit | forest | 50 | 0.993 | 0.350 |
| physics7_no_v_deficit | logreg | 50 | 0.980 | 0.050 |
| physics7_no_v_deficit | mlp | 50 | 0.980 | 0.000 |
| physics7_no_v_deficit | rules | 50 | 0.900 | 0.000 |
| physics7_no_v_deficit | tree | 50 | 0.920 | 0.000 |
| physics8 | logreg | 50 | 0.980 | 0.117 |
| physics8 | mlp | 50 | 0.980 | 0.017 |
| physics8 | forest | 50 | 0.987 | 0.000 |
| physics8 | rules | 50 | 0.900 | 0.000 |
| physics8 | tree | 50 | 0.900 | 0.000 |
| raw8 | mlp | 50 | 0.960 | 0.000 |
| raw8 | logreg | 50 | 0.960 | 0.000 |
| raw8 | tree | 50 | 0.920 | 0.000 |
| raw8 | forest | 50 | 0.907 | 0.000 |

### v_deficit ablation (pooled macro-F1)

| Model | with v_deficit | without | difference |
|---|---|---|---|
| forest | 0.975 | 0.991 | -0.0160 |
| logreg | 0.982 | 0.980 | 0.0014 |
| mlp | 0.981 | 0.980 | 0.0016 |
| rules | 0.920 | 0.920 | 0.0000 |
| tree | 0.896 | 0.911 | -0.0146 |

### v_deficit ablation (fold mean macro-F1)

| Model | with v_deficit | without | difference |
|---|---|---|---|
| forest | 0.973 | 0.992 | -0.0188 |
| logreg | 0.981 | 0.980 | 0.0016 |
| mlp | 0.981 | 0.979 | 0.0018 |
| rules | 0.919 | 0.919 | 0.0000 |
| tree | 0.881 | 0.899 | -0.0188 |

### Held-out partitions: classes present and coverage

| Fold | Runs | Classes in partition | Classes classified | Total | Classified | Skipped | Skip reasons | Runs not fully classified |
|---|---|---|---|---|---|---|---|---|
| cv0 | 10 | NOR/PAR/DUS/OPE/SHO | NOR/PAR/DUS/OPE/SHO | 600 | 600 | 0 | none | none |
| cv1 | 10 | NOR/PAR/DUS/OPE/SHO | NOR/PAR/DUS/OPE/SHO | 600 | 600 | 0 | none | none |
| cv2 | 10 | NOR/PAR/DUS/OPE/SHO | NOR/PAR/DUS/OPE/SHO | 600 | 551 | 49 | {'LUX_SATURATED': 49} | SITE-E-DUS-17 (11/60) |
| cv3 | 10 | NOR/PAR/DUS/OPE/SHO | NOR/PAR/DUS/OPE/SHO | 600 | 600 | 0 | none | none |
| cv4 | 10 | NOR/PAR/DUS/OPE/SHO | NOR/PAR/DUS/OPE/SHO | 600 | 573 | 27 | {'LUX_SATURATED': 27} | SITE-E-PAR-05 (33/60) |

## loso_training

### Reading-level metrics

| Feature set | Model | pooled macro-F1 | fold mean | fold std | fold min | pooled OPEN recall | pooled SHORT recall | size | fits |
|---|---|---|---|---|---|---|---|---|---|
| physics7_no_v_deficit | forest | 0.998 | 0.998 | 0.005 | 0.979 | 1.000 | 1.000 | 292 | 18 |
| physics7_no_v_deficit | logreg | 0.979 | 0.985 | 0.036 | 0.911 | 1.000 | 1.000 | 40 | 6 |
| physics7_no_v_deficit | mlp | 0.979 | 0.985 | 0.034 | 0.911 | 1.000 | 1.000 | 213 | 18 |
| physics7_no_v_deficit | rules | 0.951 | 0.963 | 0.073 | 0.817 | 0.998 | 1.000 | 4 | 6 |
| physics7_no_v_deficit | tree | 0.917 | 0.931 | 0.111 | 0.730 | 1.000 | 1.000 | 9 | 6 |
| physics8 | mlp | 0.980 | 0.985 | 0.034 | 0.911 | 1.000 | 1.000 | 229 | 18 |
| physics8 | logreg | 0.979 | 0.985 | 0.036 | 0.911 | 1.000 | 1.000 | 45 | 6 |
| physics8 | forest | 0.974 | 0.970 | 0.065 | 0.798 | 1.000 | 1.000 | 274 | 18 |
| physics8 | rules | 0.951 | 0.963 | 0.073 | 0.817 | 0.998 | 1.000 | 4 | 6 |
| physics8 | tree | 0.932 | 0.929 | 0.116 | 0.730 | 1.000 | 1.000 | 9 | 6 |
| raw8 | logreg | 0.931 | 0.964 | 0.059 | 0.856 | 1.000 | 1.000 | 45 | 6 |
| raw8 | mlp | 0.928 | 0.963 | 0.056 | 0.856 | 1.000 | 1.000 | 229 | 18 |
| raw8 | tree | 0.911 | 0.905 | 0.128 | 0.708 | 1.000 | 0.900 | 9 | 6 |
| raw8 | forest | 0.883 | 0.895 | 0.100 | 0.708 | 1.000 | 0.900 | 485 | 18 |

### Run-level metrics (majority vote per run, pooled over folds)

| Feature set | Model | runs | majority-vote accuracy | worst single-run reading accuracy |
|---|---|---|---|---|
| physics7_no_v_deficit | forest | 50 | 1.000 | 0.900 |
| physics7_no_v_deficit | logreg | 50 | 0.980 | 0.000 |
| physics7_no_v_deficit | mlp | 50 | 0.980 | 0.000 |
| physics7_no_v_deficit | rules | 50 | 0.940 | 0.000 |
| physics7_no_v_deficit | tree | 50 | 0.920 | 0.000 |
| physics8 | mlp | 50 | 0.980 | 0.000 |
| physics8 | logreg | 50 | 0.980 | 0.000 |
| physics8 | forest | 50 | 0.987 | 0.000 |
| physics8 | rules | 50 | 0.940 | 0.000 |
| physics8 | tree | 50 | 0.920 | 0.000 |
| raw8 | logreg | 50 | 0.940 | 0.000 |
| raw8 | mlp | 50 | 0.933 | 0.000 |
| raw8 | tree | 50 | 0.900 | 0.000 |
| raw8 | forest | 50 | 0.880 | 0.000 |

### v_deficit ablation (pooled macro-F1)

| Model | with v_deficit | without | difference |
|---|---|---|---|
| forest | 0.974 | 0.998 | -0.0238 |
| logreg | 0.979 | 0.979 | 0.0000 |
| mlp | 0.980 | 0.979 | 0.0001 |
| rules | 0.951 | 0.951 | 0.0000 |
| tree | 0.932 | 0.917 | 0.0150 |

### v_deficit ablation (fold mean macro-F1)

| Model | with v_deficit | without | difference |
|---|---|---|---|
| forest | 0.970 | 0.998 | -0.0278 |
| logreg | 0.985 | 0.985 | 0.0000 |
| mlp | 0.985 | 0.985 | 0.0001 |
| rules | 0.963 | 0.963 | 0.0000 |
| tree | 0.929 | 0.931 | -0.0016 |

### Held-out partitions: classes present and coverage

| Fold | Runs | Classes in partition | Classes classified | Total | Classified | Skipped | Skip reasons | Runs not fully classified |
|---|---|---|---|---|---|---|---|---|
| loso_SITE-A | 13 | NOR/PAR/DUS/OPE/SHO | NOR/PAR/DUS/OPE/SHO | 780 | 780 | 0 | none | none |
| loso_SITE-B | 6 | NOR/DUS/OPE | NOR/DUS/OPE | 360 | 360 | 0 | none | none |
| loso_SITE-C | 6 | NOR/OPE/SHO | NOR/OPE/SHO | 360 | 360 | 0 | none | none |
| loso_SITE-D | 9 | PAR/DUS/SHO | PAR/DUS/SHO | 540 | 540 | 0 | none | none |
| loso_SITE-E | 13 | NOR/PAR/DUS/OPE/SHO | NOR/PAR/DUS/OPE/SHO | 780 | 704 | 76 | {'LUX_SATURATED': 76} | SITE-E-DUS-17 (11/60), SITE-E-PAR-05 (33/60) |
| loso_SITE-F | 3 | DUS/OPE | DUS/OPE | 180 | 180 | 0 | none | none |

These are model-selection results on TRAINING folds. They are not the final test result (TRAINING → OPERATION), which is run once after the model is chosen.
