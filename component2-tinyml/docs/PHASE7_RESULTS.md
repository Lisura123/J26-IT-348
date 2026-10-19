# Phase 7 Results: Baseline Comparison and Model Decisions

**Status:** model-selection results on the TRAINING window only. The final TRAINING → OPERATION test has **not** been run.

Raw results: `results/baselines_phase7.json` (every fit) and `results/baselines_phase7.md` (tables). Both are unmodified outputs of `notebooks/colab_phase7_baselines.ipynb`, run in Google Colab by Member 2. Design: `docs/BASELINES.md`.

## Run record

| Item | Value |
|---|---|
| Code commit | `749351b` (Implement Baseline Comparison Framework) |
| Dataset | `pv_fault_dataset.csv`, SHA-256 `5fbb9347…5595b` (verified in the run) |
| Splits | `runwise-v1`, seed 42 |
| Environment | Google Colab: Python 3.13.16, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1 |
| Fits | 286: 2 protocols × 11 folds × 3 feature sets × 5 models; seeds 0–2 for forest and MLP |
| Warnings | 2 deprecation warnings, both from libraries (`datetime.utcnow`, SciPy L-BFGS-B options). No convergence warnings; MLP stopped after 148–236 of 2,000 iterations |
| Coverage | cv2: 49 skipped readings (`SITE-E-DUS-17`, lux saturated). cv4: 27 skipped (`SITE-E-PAR-05`). loso_SITE-E: 76 skipped. All others fully classified |

## Results (pooled macro-F1 over the combined held-out predictions)

| Feature set | Model | CV | LOSO (TRAINING) | Size |
|---|---|---|---|---|
| physics8 | **MLP 8-16-5** | **0.981** | **0.980** | 229 parameters |
| physics8 | logistic regression | 0.982 | 0.979 | 45 parameters |
| physics8 | random forest | 0.975 | 0.974 | ~264–274 nodes |
| physics8 | threshold rules | 0.920 | 0.951 | 4 thresholds |
| physics8 | decision tree | 0.896 | 0.932 | 9 nodes |
| physics7 (no `v_deficit`) | random forest | 0.991 | 0.998 | ~278–292 nodes |
| raw8 | MLP 8-16-5 | 0.958 | 0.928 | 229 parameters |
| raw8 | logistic regression | 0.957 | 0.931 | 45 parameters |

- Run-level majority-vote accuracy for the physics8 MLP is 0.980 in both protocols (50 runs).
- Pooled OPEN_CIRCUIT and SHORT_CIRCUIT recall is 1.000 for all physics8 ML models.

## Findings

1. **PV-physics features outperform the raw sensor features.** Pooled macro-F1 for the MLP is +2.3 points under CV and +5.2 points under leave-one-site-out. Logistic regression shows the same pattern. The largest gain is on unseen sites.
2. **The MLP is not more accurate than logistic regression.** The two are within 0.001 of each other in both protocols.
3. **`v_deficit` has no demonstrated meaningful benefit for the MLP.** The ablation (with vs without `v_deficit`, pooled macro-F1) gives:

   | Model | CV difference | LOSO difference |
   |---|---|---|
   | MLP | +0.0016 | +0.0001 |
   | logistic regression | +0.0014 | 0.0000 |
   | random forest | −0.016 | −0.024 |
   | decision tree | −0.015 | +0.015 |

   The feature neither helps nor harms the MLP and logistic regression on this data. It reduces forest performance, which is consistent with the cross-site voltage-reference bias noted in `FEATURES.md` (Finding B).
4. **ML models outperform the threshold rules by about 3–6 points.** The rules misclassify low-light NORMAL and DUST runs (~5,000 lx) as OPEN_CIRCUIT (`SITE-A-NOR-19`, `SITE-E-DUS-19`) and miss partial shading.
5. **False OPEN_CIRCUIT predictions on a shading run.** Logistic regression and the MLP classify 53–60 readings of `SITE-A-PAR-01` (PARTIAL_SHADING) as OPEN_CIRCUIT. That run has a current of about 18.6 mA, whereas OPEN_CIRCUIT runs have a median of 0.1 mA, and it shows no hotspot (median |T1−T2| 0.86 °C vs 2.1 °C for shading). The models react to its low current per lux. Fault recall is not affected, but this is a false-alarm risk.
6. **PARTIAL_SHADING is the main remaining error.** Recall is about 0.90 for the best models; the other classes are at about 1.0.

## Decisions (approved by Member 2)

1. **Feature set: physics8, including `v_deficit`.** Kept because it does not harm the MLP and preserves the 8-input, 229-parameter layout that Member 3's federated learning uses. `v_deficit` has **no demonstrated meaningful performance benefit** for the MLP on this dataset, and it is not claimed as a contribution.
2. **Proposed deployment model: 8-16-5 tanh MLP (229 parameters) on physics8.** The selection is based on the overall system architecture and **federated learning compatibility**, not on superiority over logistic regression, which performs equally well. Logistic regression stays as the reported baseline.
3. **Electrical fault confirmation layer (planned safety mechanism).** In the deployment design, OPEN_CIRCUIT and SHORT_CIRCUIT predictions must pass electrical validation before any automatic isolation decision. For example, OPEN requires near-zero current and SHORT requires near-zero voltage; thresholds are to be set with Member 1's hardware requirements. Finding 5 motivates this. **It is a planned design element and has not been implemented or validated.** Relay control remains a proposal (`ARCHITECTURE.md` §7.5).

## Limitations
- TRAINING window only. The final OPERATION test has not been run.
- Scores are close to the ceiling (~0.98). That points to a dataset that is easy to separate. Dataset provenance remains unresolved (`DATA_PROVENANCE.md`).
- Results were produced with scikit-learn 1.6.1. Other versions may give slightly different numbers.
- The LOSO folds use TRAINING runs only and lack some classes at some sites (see the class table in `results/baselines_phase7.md`).
- Model size is a parameter or node count. It is not a measured ESP32 flash, RAM or latency figure.
