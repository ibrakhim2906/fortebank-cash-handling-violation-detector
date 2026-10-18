import pytest
from scripts.mission_validation import alias_number,seconds


def test_first_three_column_adapter_aliases_and_times():
    assert alias_number('X-1.mov')==1
    assert alias_number('Х-47')==47
    assert alias_number('#U0425-12.mov')==12
    assert seconds('1минута 8 секунда')==68
    assert seconds('22 секунда')==22
    assert seconds(float('nan')) is None
    with pytest.raises(ValueError):seconds('unknown')
    with pytest.raises(ValueError):alias_number('unrelated-video')
