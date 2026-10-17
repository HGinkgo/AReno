import pytest

from areno.engine.inference import _cancel_stop_token


@pytest.mark.parametrize("stops,eos,expected", [([], (), 0), ([], None, 0), ([], (7, 8), 7), ([], 9, 9), ([3], (), 3)])
def test_cancel_stop_token_accepts_ignore_eos_empty_tuple(stops, eos, expected):
    assert _cancel_stop_token(stops, eos) == expected
