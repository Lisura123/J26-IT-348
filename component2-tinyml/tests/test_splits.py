import json

import numpy as np
import pandas as pd
import pytest

from tinyml.contract import LABELS
from tinyml.splits import iter_folds, load_assignments, make_assignments, summarize, to_json

CODES = {"NORMAL": "NOR", "PARTIAL_SHADING": "PAR", "DUST": "DUS",
         "OPEN_CIRCUIT": "OPE", "SHORT_CIRCUIT": "SHO"}
SITES = ["SITE-A", "SITE-B", "SITE-C"]


def make_runs(runs_per_window=5, readings=3, skip=(("SITE-C", "DUST"),)):
    """Small dataset: per label, `runs_per_window` runs in each window, spread
    over three sites. Pairs in `skip` (site, label) get no runs, to mimic
    sites that lack a class."""
    rows, rec = [], 0
    for label in LABELS:
        sites = [s for s in SITES if (s, label) not in skip]
        n = 0
        for window in ("TRAINING", "OPERATION"):
            for r in range(runs_per_window):
                n += 1
                site = sites[n % len(sites)]
                run = f"{site}-{CODES[label]}-{n:02d}"
                for k in range(readings):
                    rec += 1
                    rows.append({"record_id": f"REC-{rec:05d}", "run_id": run, "site_id": site,
                                 "window": window, "label": label, "reading": k + 1,
                                 "voltage_v": 2.5, "health_score": 50.0})
    return pd.DataFrame(rows)


@pytest.fixture
def df():
    return make_runs()


@pytest.fixture
def assignments(df):
    return make_assignments(df, n_splits=5, seed=42)


@pytest.mark.parametrize("protocol", ["cv", "temporal", "loso"])
def test_no_run_in_train_and_test(df, assignments, protocol):
    for name, tr, te in iter_folds(df, assignments, protocol):
        assert not set(df.loc[tr, "run_id"]) & set(df.loc[te, "run_id"]), name
        assert not set(tr) & set(te), name


def test_every_run_has_one_assignment(df, assignments):
    assert set(assignments.index) == set(df.run_id)
    assert assignments.index.is_unique
    assert set(assignments.temporal) == {"train", "test"}


def test_cv_covers_each_training_record_exactly_once(df, assignments):
    seen = np.concatenate([te for _, _, te in iter_folds(df, assignments, "cv")])
    training = df.index[df.window == "TRAINING"]
    assert sorted(seen) == sorted(training)


def test_cv_uses_only_training_runs(df, assignments):
    for _, tr, te in iter_folds(df, assignments, "cv"):
        assert set(df.loc[tr, "window"]) == {"TRAINING"}
        assert set(df.loc[te, "window"]) == {"TRAINING"}
    assert (assignments.loc[assignments.window == "OPERATION", "cv_fold"] == -1).all()


def test_loso_covers_every_record_once(df, assignments):
    seen = np.concatenate([te for _, _, te in iter_folds(df, assignments, "loso")])
    assert sorted(seen) == sorted(df.index)


def test_same_seed_is_reproducible(df):
    a = make_assignments(df, seed=7)
    b = make_assignments(df.sample(frac=1, random_state=1), seed=7)   # row order must not matter
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_cv_folds(df):
    a = make_assignments(df, seed=1)
    b = make_assignments(df, seed=2)
    assert not a.cv_fold.equals(b.cv_fold)


def test_cv_folds_are_stratified(df, assignments):
    cv = assignments[assignments.cv_fold >= 0]
    counts = pd.crosstab(cv.cv_fold, cv.label)
    assert (counts == 1).all().all()            # 5 runs per class over 5 folds -> one each


def test_temporal_separation(df, assignments):
    (_, tr, te), = iter_folds(df, assignments, "temporal")
    assert set(df.loc[tr, "window"]) == {"TRAINING"}
    assert set(df.loc[te, "window"]) == {"OPERATION"}
    assert len(tr) + len(te) == len(df)


def test_site_separation(df, assignments):
    folds = list(iter_folds(df, assignments, "loso"))
    assert [n for n, _, _ in folds] == [f"loso_{s}" for s in SITES]
    for name, tr, te in folds:
        site = name.removeprefix("loso_")
        assert set(df.loc[te, "site_id"]) == {site}
        assert site not in set(df.loc[tr, "site_id"])


def test_missing_class_at_site_is_reported(df, assignments):
    loso = {f["fold"]: f for f in summarize(df, assignments)["loso"]}
    assert loso["loso_SITE-C"]["test_missing"] == ["DUST"]
    assert loso["loso_SITE-A"]["test_missing"] == []
    assert loso["loso_SITE-C"]["train_runs"]["DUST"] > 0     # still learnable from other sites


def test_too_few_runs_for_cv_raises():
    small = make_runs(runs_per_window=3)
    with pytest.raises(ValueError, match="runs per class"):
        make_assignments(small, n_splits=5)


def test_run_with_two_labels_raises(df):
    df.loc[0, "label"] = "DUST" if df.loc[0, "label"] != "DUST" else "NORMAL"
    with pytest.raises(ValueError, match="more than one label"):
        make_assignments(df)


def test_unassigned_run_is_rejected(df, assignments):
    with pytest.raises(ValueError, match="no split assignment"):
        list(iter_folds(df, assignments.drop(assignments.index[0]), "cv"))


def test_split_file_round_trip(tmp_path, df, assignments):
    path = tmp_path / "splits.json"
    path.write_text(to_json(assignments, {"dataset_sha256": "abc", "seed": 42}))
    loaded, meta = load_assignments(path, df, dataset_sha256="abc")
    assert meta["seed"] == 42
    assert loaded.cv_fold.astype(int).equals(assignments.cv_fold.astype(int))
    with pytest.raises(ValueError, match="different dataset"):
        load_assignments(path, df, dataset_sha256="other")


def test_split_file_does_not_store_features(df, assignments):
    run = json.loads(to_json(assignments, {}))["runs"][assignments.index[0]]
    assert "health_score" not in run and "voltage_v" not in run
