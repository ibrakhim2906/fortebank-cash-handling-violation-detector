"""Operational variation gates use synthetic observations, never video targets."""
import numpy as np
import pytest

from scripts.check_spatial_pilot import assess


FIELDS = ("people_count", "visible_hands", "hand_object_contact", "rectangular_sheets_or_stacks",
          "handheld_device", "open_container", "work_surface", "visibility")


def records(*, constant=False, unclear=False, slow=False):
    result = []
    for index in range(3):
        observation = {field: "unclear" if unclear else "visible" for field in FIELDS}
        observation.update(visibility="partial", people_count="one", visible_hands="two")
        if index and not constant:
            observation.update(people_count="two", handheld_device="not_visible")
        result.append({"clip_features": np.full(84, 0 if constant else index, dtype=float),
                       "observation_audit": [{"observation": dict(observation)} for _ in range(4)],
                       "extraction_sec": 181 if slow else 40})
    return result


def test_variation_and_definite_coverage_pass_without_any_target_input():
    report = assess(records(), FIELDS)
    assert report["passed"] is True
    assert report["valid_windows"] == 12
    assert report["varying_definite_fields"] == 2
    assert report["definite_object_fraction"] == 1
    assert report["binary_labels_used"] is False
    assert report["semantic_accuracy_verified"] is False


@pytest.mark.parametrize("options", [{"constant": True}, {"unclear": True}, {"slow": True}])
def test_constant_unknown_or_slow_pilot_cannot_start_full_extraction(options):
    assert assess(records(**options), FIELDS)["passed"] is False


def test_visibility_change_alone_cannot_pass_the_observation_gate():
    items = records(constant=True)
    for index, item in enumerate(items):
        item["clip_features"] = np.full(84, index)
        for sample in item["observation_audit"]:
            sample["observation"]["visibility"] = "clear" if index else "partial"
    assert assess(items, FIELDS)["passed"] is False


def test_missing_pilot_window_cannot_pass():
    items = records()
    items[0]["observation_audit"].pop()
    assert assess(items, FIELDS)["passed"] is False
