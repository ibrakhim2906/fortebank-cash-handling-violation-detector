"""Template leakage test: features must not change when events after the cutoff are deleted.
Copy this pattern for every new feature family.  Run: uv run pytest"""
import pandas as pd

from src.features import agg_windows


def test_agg_windows_ignores_future_events():
    ev = pd.DataFrame({
        "cid": [1, 1, 1, 2, 2],
        "d": pd.to_datetime(["2026-01-01", "2026-02-20", "2026-03-10", "2026-02-25", "2026-04-01"]),
        "amt": [100, 200, 300, 50, 999],
    })
    ref = pd.DataFrame({"cid": [1, 2], "ref_date": pd.to_datetime(["2026-03-01", "2026-03-01"])})
    with_future = agg_windows(ev, "cid", "d", ["amt"], ref)
    without_future = agg_windows(ev[ev["d"] < "2026-03-01"], "cid", "d", ["amt"], ref)
    pd.testing.assert_frame_equal(with_future, without_future)
