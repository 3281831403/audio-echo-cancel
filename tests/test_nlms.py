import math

import pytest

from aec.nlms import (
    AecError,
    align_ref,
    cancel_batch,
    delay_estimate,
    double_talk_guard,
    erle,
    estimate_echo,
    nlms_step,
    validate_params,
)

TOL = 1e-6


def close(value):
    return pytest.approx(value, abs=TOL, rel=0.0)


def close_list(values):
    return pytest.approx(values, abs=TOL, rel=0.0)


def test_delay_estimate_reports_peak_lag_not_window_index():
    ref = [0.0] * 24
    ref[4] = 1.0
    mic = [0.0] * 24
    mic[11] = 1.0

    assert delay_estimate(mic, ref, 12) == 7
    assert delay_estimate(mic, ref, -1) == -1


def test_align_ref_pads_before_reference_not_after():
    ref = [1.0, 2.0, 3.0, 4.0]

    assert align_ref(ref, 2) == close_list([0.0, 0.0, 1.0, 2.0])
    assert align_ref(ref, 0) == close_list(ref)


def test_nlms_step_normalizes_update_by_reference_energy():
    updated = nlms_step([0.0, 0.0], [1000.0, 0.0], 1.0, 1.0)

    assert updated == close_list([0.001, 0.0])
    assert all(math.isfinite(value) for value in updated)


def test_nlms_step_moves_weights_along_negative_error_gradient():
    updated = nlms_step([0.0, 0.0], [1.0, 1.0], 2.0, 0.5)

    assert updated == close_list([0.5, 0.5])


def test_estimate_echo_multiplies_weights_by_reference():
    assert estimate_echo([1.0, -2.0, 0.5], [4.0, 3.0, 2.0]) == close(-1.0)


def test_erle_returns_db_not_linear_ratio():
    assert erle([1.0, 1.0], [0.1, 0.1]) == close(20.0)


def test_double_talk_guard_compares_error_energy_with_mic_energy():
    assert double_talk_guard([10.0, 0.0], [1.0, 0.0], 0.5) is False
    assert double_talk_guard([10.0, 0.0], [9.0, 0.0], 0.5) is True


def test_validate_params_rejects_out_of_range_step():
    problems = validate_params(16, 3.0)

    assert problems
    assert any("mu" in problem.lower() for problem in problems)
    assert validate_params(16, 0.5) == []


def test_cancel_batch_keeps_channel_weights_independent():
    config = {
        "length": 4,
        "mu": 0.5,
        "max_delay": 3,
        "double_talk_threshold": 0.5,
    }
    first = ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    second = ([0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 1.0, 0.0])

    solo = cancel_batch([second], config)
    batch = cancel_batch([first, second], config)

    assert batch["weights"][1] == close_list(solo["weights"][0])
    assert batch["residuals"][1] == close_list(solo["residuals"][0])


def test_validate_params_rejects_non_positive_length():
    problems = validate_params(0, 0.5)

    assert problems
    assert any("length" in problem.lower() for problem in problems)


def test_zero_echo_residual_equals_microphone():
    mic = [0.3, -0.7, 0.0, 1.2]
    ref = [0.0] * len(mic)

    result = cancel_batch([(mic, ref)], {"length": 4, "mu": 0.5})

    assert result["residuals"][0] == close_list(mic)


def test_zero_mu_keeps_weights_unchanged():
    weights = [0.25, -0.5]

    updated = nlms_step(weights, [1.0, 2.0], 3.0, 0.0)

    assert updated == close_list(weights)
    assert weights == close_list([0.25, -0.5])


def test_silence_erle_is_infinite():
    assert erle([0.0] * 16, [0.0] * 16) == math.inf


def test_zero_length_batch_raises_aec_error():
    with pytest.raises(AecError):
        cancel_batch([([1.0, 0.0], [1.0, 0.0])], {"length": 0, "mu": 0.5})


def test_cancel_batch_preserves_success_order():
    pairs = [
        ([0.2, -0.4], [0.0, 0.0]),
        ([0.1, 0.2, 0.3], [0.0, 0.0, 0.0]),
    ]

    result = cancel_batch(pairs, {"length": 2, "mu": 0.5})

    assert [len(residual) for residual in result["residuals"]] == [2, 3]
    assert result["delays"] == [0, 0]
