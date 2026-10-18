import numpy as np
import pandas as pd
import pytest

from src import config
from src.eval import make_folds


@pytest.fixture(autouse=True)
def validation_config(monkeypatch):
    settings = {
        "TIME_COL": "date",
        "GROUP_COL": None,
        "TARGET": "target",
        "TASK": "binary",
        "N_FOLDS": 3,
        "LABEL_GAP_DAYS": 0,
        "VALIDATION_WINDOWS": None,
        "HOLDOUT_WINDOW": None,
        "EXCLUDE_VALIDATION_GROUPS": False,
    }
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)


@pytest.fixture
def applications():
    days = np.arange(24)
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=24).repeat(2),
        "client": np.column_stack((days, days + 1)).ravel(),
        "target": np.tile([0, 1], 24),
        "row_id": np.arange(48),
    }, index=np.arange(100, 148)).sample(frac=1, random_state=config.SEED)


def assert_time_folds(frame, folds, gap=0, exclude_groups=False):
    dates = pd.to_datetime(frame[config.TIME_COL]).dt.normalize()
    for training, validation in folds:
        train_dates = dates.iloc[training]
        valid_dates = dates.iloc[validation]
        assert len(training) and len(validation)
        assert not set(training) & set(validation)
        assert not set(train_dates) & set(valid_dates)
        assert train_dates.is_monotonic_increasing
        assert valid_dates.is_monotonic_increasing
        assert (train_dates + pd.Timedelta(days=gap) < valid_dates.min()).all()
        if exclude_groups:
            assert not set(frame.iloc[training][config.GROUP_COL]) & set(
                frame.iloc[validation][config.GROUP_COL]
            )


def test_automatic_windows_split_unique_dates(applications):
    folds = make_folds(applications)
    dates = applications.date
    chunks = np.array_split(dates.sort_values().unique(), config.N_FOLDS + 1)
    assert len(folds) == config.N_FOLDS
    for fold_index, (training, validation) in enumerate(folds):
        assert set(dates.iloc[validation]) == set(chunks[fold_index + 1])
        assert set(dates.iloc[training]) == set(np.concatenate(chunks[:fold_index + 1]))
    assert_time_folds(applications, folds)


def test_uneven_date_counts_do_not_split_a_day(applications):
    extra = applications[applications.date == "2025-01-08"]
    uneven = pd.concat([applications, extra, extra, extra], ignore_index=True)
    folds = make_folds(uneven)
    assert_time_folds(uneven, folds)
    for _, validation in folds:
        valid_dates = set(uneven.iloc[validation].date)
        assert set(validation) == set(np.flatnonzero(uneven.date.isin(valid_dates)))


def test_intraday_timestamps_share_date_boundaries(applications):
    frame = applications.copy()
    frame["date"] += pd.to_timedelta(np.where(frame.row_id % 2, 17, 8), unit="h")
    folds = make_folds(frame)
    assert_time_folds(frame, folds)
    assert len(folds) == config.N_FOLDS


@pytest.mark.parametrize("gap", [0, 2])
@pytest.mark.parametrize("automatic", [False, True])
def test_strict_gap_boundary(applications, monkeypatch, gap, automatic):
    monkeypatch.setattr(config, "LABEL_GAP_DAYS", gap)
    if not automatic:
        monkeypatch.setattr(config, "VALIDATION_WINDOWS", [("2025-01-07", "2025-01-13")])
    folds = make_folds(applications)
    for training, validation in folds:
        cutoff = applications.iloc[validation].date.min() - pd.Timedelta(days=gap)
        assert set(training) == set(np.flatnonzero(applications.date < cutoff))
        assert not (applications.iloc[training].date == cutoff).any()
    if not automatic:
        expected = applications.date.ge("2025-01-07") & applications.date.lt("2025-01-13")
        assert set(folds[0][1]) == set(np.flatnonzero(expected))
    assert_time_folds(applications, folds, gap=gap)


@pytest.mark.parametrize("exclude", [False, True])
def test_group_exclusion_is_opt_in(applications, monkeypatch, exclude):
    monkeypatch.setattr(config, "GROUP_COL", "client")
    monkeypatch.setattr(config, "EXCLUDE_VALIDATION_GROUPS", exclude)
    folds = make_folds(applications)
    for training, validation in folds:
        shared = set(applications.iloc[training].client) & set(applications.iloc[validation].client)
        assert bool(shared) is not exclude
    assert_time_folds(applications, folds, exclude_groups=exclude)


def test_time_mode_works_without_groups(applications):
    assert config.GROUP_COL is None
    frame = applications.drop(columns="client")
    assert_time_folds(frame, make_folds(frame))


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("holdout_window", [
    ("2025-01-19", "2025-01-25"),
    ("2025-01-09", "2025-01-11"),
])
def test_holdout_excluded_from_all_development_rows(
    applications, monkeypatch, automatic, holdout_window
):
    monkeypatch.setattr(config, "HOLDOUT_WINDOW", holdout_window)
    if not automatic:
        monkeypatch.setattr(config, "VALIDATION_WINDOWS", [
            ("2025-01-07", "2025-01-13"),
            ("2025-01-13", "2025-01-19"),
        ])
    start, end = map(pd.Timestamp, holdout_window)
    reserved = set(np.flatnonzero((applications.date >= start) & (applications.date < end)))
    folds = make_folds(applications)
    for training, validation in folds:
        assert not reserved & set(training)
        assert not reserved & set(validation)
    assert_time_folds(applications, folds)
    holdout_folds = make_folds(applications, holdout=True)
    assert len(holdout_folds) == 1
    assert set(holdout_folds[0][1]) == reserved
    assert_time_folds(applications, holdout_folds)


def test_holdout_respects_gap_and_group_exclusion(applications, monkeypatch):
    monkeypatch.setattr(config, "GROUP_COL", "client")
    monkeypatch.setattr(config, "EXCLUDE_VALIDATION_GROUPS", True)
    monkeypatch.setattr(config, "LABEL_GAP_DAYS", 2)
    monkeypatch.setattr(config, "HOLDOUT_WINDOW", ("2025-01-19", "2025-01-25"))
    assert_time_folds(applications, make_folds(applications, holdout=True), gap=2, exclude_groups=True)


@pytest.mark.parametrize("task", ["binary", "regression"])
@pytest.mark.parametrize("group_column", [None, "client"])
def test_no_time_column(applications, monkeypatch, task, group_column):
    monkeypatch.setattr(config, "TIME_COL", None)
    monkeypatch.setattr(config, "TASK", task)
    monkeypatch.setattr(config, "GROUP_COL", group_column)
    frame = applications.drop(columns="date")
    folds = make_folds(frame)
    assert len(folds) == config.N_FOLDS
    assert sorted(np.concatenate([validation for _, validation in folds])) == list(range(len(frame)))
    for training, validation in folds:
        assert not set(training) & set(validation)
        if group_column:
            assert not set(frame.iloc[training].client) & set(frame.iloc[validation].client)


def test_membership_is_independent_of_input_order(applications):
    sorted_frame = applications.sort_values("date").reset_index(drop=True)
    for original, reordered in zip(make_folds(applications), make_folds(sorted_frame)):
        for original_indices, reordered_indices in zip(original, reordered):
            assert set(applications.iloc[original_indices].row_id) == set(
                sorted_frame.iloc[reordered_indices].row_id
            )


@pytest.mark.parametrize("settings, message", [
    ({"EXCLUDE_VALIDATION_GROUPS": True}, "requires GROUP_COL"),
    ({"LABEL_GAP_DAYS": -1}, "non-negative integer"),
    ({"LABEL_GAP_DAYS": 1.5}, "non-negative integer"),
    ({"N_FOLDS": 30}, "unique dates"),
    ({"N_FOLDS": 0}, "unique dates"),
    ({"VALIDATION_WINDOWS": []}, "No validation windows"),
    ({"VALIDATION_WINDOWS": [("2025-01-07", "2025-01-07")]}, "start < end"),
    ({"VALIDATION_WINDOWS": ["2025-01-07"]}, "start and exclusive end"),
    ({"VALIDATION_WINDOWS": [("2025-01-07", "2025-01-13"),
                             ("2025-01-12", "2025-01-19")]}, "non-overlapping"),
    ({"LABEL_GAP_DAYS": 100}, "Empty training or validation"),
])
def test_invalid_settings_fail_clearly(applications, monkeypatch, settings, message):
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)
    with pytest.raises(ValueError, match=message):
        make_folds(applications)


def test_missing_dates_fail(applications):
    applications.loc[applications.index[0], "date"] = pd.NaT
    with pytest.raises(ValueError, match="dates must be complete"):
        make_folds(applications)


def test_missing_excluded_groups_fail(applications, monkeypatch):
    monkeypatch.setattr(config, "GROUP_COL", "client")
    monkeypatch.setattr(config, "EXCLUDE_VALIDATION_GROUPS", True)
    applications["client"] = applications.client.astype("Int64")
    applications.loc[applications.index[0], "client"] = pd.NA
    with pytest.raises(ValueError, match="groups must be complete"):
        make_folds(applications)


def test_holdout_requires_configuration(applications):
    with pytest.raises(ValueError, match="requires HOLDOUT_WINDOW"):
        make_folds(applications, holdout=True)


def test_holdout_requires_time(applications, monkeypatch):
    monkeypatch.setattr(config, "TIME_COL", None)
    monkeypatch.setattr(config, "HOLDOUT_WINDOW", ("2025-01-19", "2025-01-25"))
    with pytest.raises(ValueError, match="requires TIME_COL"):
        make_folds(applications)


@pytest.mark.parametrize("automatic", [False, True])
def test_mock_fold_table(monkeypatch, capsys, automatic):
    path = config.DATA_RAW / "applications_train.csv"
    if not path.exists():
        pytest.skip("Optional mock raw data is not available in this checkout")
    frame = pd.read_csv(path)
    monkeypatch.setattr(config, "TIME_COL", "application_date")
    monkeypatch.setattr(config, "GROUP_COL", "client_id")
    monkeypatch.setattr(config, "N_FOLDS", 2)
    monkeypatch.setattr(config, "HOLDOUT_WINDOW", ("2026-02-01", "2026-03-01"))
    if not automatic:
        monkeypatch.setattr(config, "LABEL_GAP_DAYS", 183)
        monkeypatch.setattr(config, "EXCLUDE_VALIDATION_GROUPS", True)
        monkeypatch.setattr(config, "VALIDATION_WINDOWS", [
            ("2025-12-01", "2026-01-01"),
            ("2026-01-01", "2026-02-01"),
        ])
    folds = make_folds(frame)
    holdout_folds = make_folds(frame, holdout=True)
    assert len(folds) == 2
    assert len(holdout_folds) == 1
    if not automatic:
        assert [(len(training), len(validation)) for training, validation in folds + holdout_folds] == [
            (3066, 1046), (4167, 1064), (5181, 890),
        ]
    assert_time_folds(frame, folds + holdout_folds, gap=config.LABEL_GAP_DAYS,
                      exclude_groups=config.EXCLUDE_VALIDATION_GROUPS)
    dates = pd.to_datetime(frame.application_date)
    for training, validation in folds:
        assert (dates.iloc[training] < "2026-02-01").all()
        assert (dates.iloc[validation] < "2026-02-01").all()
    assert dates.iloc[holdout_folds[0][1]].min() == pd.Timestamp("2026-02-01")
    assert dates.iloc[holdout_folds[0][1]].max() == pd.Timestamp("2026-02-28")
    rows = []
    for fold_index, (training, validation) in enumerate(folds + holdout_folds):
        earlier, later = frame.iloc[training], frame.iloc[validation]
        rows.append({
            "fold": "holdout" if fold_index == len(folds) else f"dev {fold_index + 1}",
            "train dates": f"{earlier.application_date.min()}..{earlier.application_date.max()}",
            "valid dates": f"{later.application_date.min()}..{later.application_date.max()}",
            "train rows": len(training),
            "valid rows": len(validation),
            "shared clients": len(set(earlier.client_id) & set(later.client_id)),
        })
    with capsys.disabled():
        print(f"\nMock folds: automatic={automatic}, gap={config.LABEL_GAP_DAYS} days, "
              f"exclude groups={config.EXCLUDE_VALIDATION_GROUPS}")
        print(pd.DataFrame(rows).to_string(index=False))
