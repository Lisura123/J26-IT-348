# Run-wise split summary

Split version `runwise-v1`, seed 42, 5-fold CV over TRAINING runs. Dataset SHA-256 `5fbb93471c056c274cbf81e8c317934e379b2484013147e4c2cacfa93421595b`.

Every run is kept whole: no `run_id` appears in both train and test of any split. `run_id` and `site_id` are grouping keys only, never model features.

Cells show runs per class as NOR/PAR/DUS/OPE/SHO.

## Protocol 1: run-wise stratified CV (model selection, TRAINING window only)

| Fold | Train runs | Train rows | Test runs | Test rows | Classes missing in test |
|---|---|---|---|---|---|
| cv0 | 8/8/8/8/8 | 2400 | 2/2/2/2/2 | 600 | none |
| cv1 | 8/8/8/8/8 | 2400 | 2/2/2/2/2 | 600 | none |
| cv2 | 8/8/8/8/8 | 2400 | 2/2/2/2/2 | 600 | none |
| cv3 | 8/8/8/8/8 | 2400 | 2/2/2/2/2 | 600 | none |
| cv4 | 8/8/8/8/8 | 2400 | 2/2/2/2/2 | 600 | none |

## Protocol 2: TRAINING → OPERATION (final held-out test, used once)

| Fold | Train runs | Train rows | Test runs | Test rows | Classes missing in test |
|---|---|---|---|---|---|
| temporal | 10/10/10/10/10 | 3000 | 10/10/10/10/10 | 3000 | none |

## Protocol 3: leave-one-site-out (generalisation to an unseen site, all runs)

| Fold | Train runs | Train rows | Test runs | Test rows | Classes missing in test |
|---|---|---|---|---|---|
| loso_SITE-A | 14/10/18/19/18 | 4740 | 6/10/2/1/2 | 1260 | none |
| loso_SITE-B | 15/20/14/18/19 | 5160 | 5/0/6/2/1 | 840 | PAR |
| loso_SITE-C | 17/20/19/12/15 | 4980 | 3/0/1/8/5 | 1020 | PAR |
| loso_SITE-D | 18/13/17/20/16 | 5040 | 2/7/3/0/4 | 960 | OPE |
| loso_SITE-E | 16/18/13/15/12 | 4440 | 4/2/7/5/8 | 1560 | none |
| loso_SITE-F | 20/19/19/16/20 | 5640 | 0/1/1/4/0 | 360 | NOR, SHO |

## Limitations

- **CV** uses only the 50 TRAINING runs (10 per class, 2 per class per fold), so the OPERATION runs stay unseen until the final test.
- **Temporal** has one split only; its result is a single estimate (report per-class recall and run-level results, not only accuracy).
- **Leave-one-site-out:** some sites do not contain every class, so their test folds cannot measure recall for those classes. Report macro-F1 over the classes present in each fold, list the missing classes, and give the pooled result over all six folds. Site sizes differ (6 to 26 runs), so folds are unequal; SITE-F (6 runs) is the least reliable fold.
- LOSO mixes TRAINING and OPERATION runs and is a separate generalisation study; it is not used for model selection.
