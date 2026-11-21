"""Normalized LMS building blocks for acoustic echo cancellation.

All signal frames are plain ``list[float]`` values.  Floating-point results
in this module are defined with an absolute tolerance of ``1e-6``.
"""

import math

# Floor used both to avoid division by zero on a silent reference frame and
# to keep an active step size strictly inside the open interval (0, 2).
_EPS = 1e-12


class AecError(ValueError):
    """Raised when echo-cancellation parameters or a batch are invalid."""


def delay_estimate(mic: list[float], ref: list[float], max_delay: int) -> int:
    """Estimate the integer delay of ``ref`` relative to ``mic``.

    The returned delay ``d`` satisfies ``mic[t] ~= ref[t - d]`` for the best
    non-negative lag in ``[0, max_delay]``.  Score each lag with the
    cross-correlation ``sum(mic[t] * ref[t - d])`` over overlapping samples
    and return the lag with the largest positive score; ties resolve to the
    smaller lag.

    Return ``-1`` when ``max_delay < 0``, when ``max_delay`` is not smaller
    than both signal lengths, or when either signal is empty.  If every
    candidate score is zero (including two silent frames), return ``0``.
    """
    if not isinstance(max_delay, int) or max_delay < 0:
        return -1

    n_mic = len(mic)
    n_ref = len(ref)
    if n_mic == 0 or n_ref == 0:
        return -1
    # The lag must be strictly smaller than both signal lengths so that
    # every candidate lag still has overlapping samples.
    if max_delay >= n_mic or max_delay >= n_ref:
        return -1

    best_lag = 0
    best_score = 0.0
    for lag in range(max_delay + 1):
        score = 0.0
        # mic[t] pairs with ref[t - lag] for t in [lag, lag + n_ref),
        # clipped to the microphone's extent.
        upper = min(n_mic, lag + n_ref)
        for t in range(lag, upper):
            score += mic[t] * ref[t - lag]
        # Strict comparison keeps the smallest lag on a tie and only lets
        # a genuinely positive score replace the zero-score default.
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def align_ref(ref: list[float], delay: int) -> list[float]:
    """Delay ``ref`` by ``delay`` samples, padding with leading zeros.

    The result has the same length as ``ref``: its first ``delay`` samples
    are ``0.0`` and the remaining samples are ``ref[:len(ref) - delay]``.
    Samples shifted past the end are discarded.  A non-positive delay
    returns a copy of ``ref``; a delay greater than or equal to ``len(ref)``
    returns an all-zero list.  The result must not alias ``ref``.
    """
    n = len(ref)
    if delay <= 0:
        return list(ref)
    if delay >= n:
        return [0.0] * n
    # Leading zeros delay the reference; the tail is truncated, not wrapped.
    return [0.0] * delay + list(ref[: n - delay])


def nlms_step(
    weights: list[float],
    x: list[float],
    e: float,
    mu: float,
) -> list[float]:
    """Return the weights after one normalized-LMS update.

    With reference-frame energy ``g = sum(x[i] * x[i])``, the update is::

        w_new[i] = w[i] + mu_eff * e * x[i] / (g + 1e-12)

    ``e`` is the current residual (desired signal minus estimated echo).
    Active step sizes are clamped to the open interval ``(0, 2)`` so that
    large reference energy or an oversized ``mu`` cannot make the update
    diverge.  ``mu == 0`` is the explicit freeze case: the weights are
    returned unchanged, in a new list.  Negative ``mu`` also freezes.
    Finite inputs must produce finite outputs, and the arguments must not be
    mutated.
    """
    updated = list(weights)

    # Freeze: zero or negative step sizes perform no adaptation.
    if mu <= 0:
        return updated

    # Clamp into the open interval (0, 2) so mu >= 2 cannot diverge.
    mu_eff = mu
    if mu_eff >= 2.0:
        mu_eff = 2.0 - _EPS

    # Normalize by the reference energy.  The epsilon floor means a silent
    # frame yields a zero update instead of dividing by zero.
    energy = 0.0
    for sample in x:
        energy += sample * sample
    denom = energy + _EPS

    for i in range(len(updated)):
        # Plus sign: move along the negative gradient, i.e. toward reducing
        # the residual, never amplifying it.
        updated[i] = updated[i] + mu_eff * e * x[i] / denom
    return updated


def estimate_echo(weights: list[float], x: list[float]) -> float:
    """Return the current echo estimate for one frame.

    ``weights`` and ``x`` must have the same length; the result is their dot
    product, ``sum(weights[i] * x[i])``.  An empty frame yields ``0.0``.
    """
    total = 0.0
    for w, sample in zip(weights, x):
        total += w * sample
    return total


def erle(mic: list[float], residual: list[float]) -> float:
    """Return echo-return-loss enhancement in dB.

    The value is ``10 * log10(mic_energy / residual_energy)``, where each
    energy is the sum of squared samples in that frame.  Digital silence in
    both frames (zero microphone and zero residual energy) returns
    ``math.inf``.  A zero residual with a non-zero microphone also returns
    ``math.inf``; a zero microphone with a non-zero residual returns
    ``-math.inf``.  Frame lengths may differ.
    """
    mic_energy = 0.0
    for sample in mic:
        mic_energy += sample * sample
    residual_energy = 0.0
    for sample in residual:
        residual_energy += sample * sample

    if residual_energy == 0.0:
        # Covers both the all-silent case and perfect cancellation.
        return math.inf
    if mic_energy == 0.0:
        return -math.inf
    return 10.0 * math.log10(mic_energy / residual_energy)


def double_talk_guard(
    mic: list[float],
    err: list[float],
    threshold: float,
) -> bool:
    """Return whether near-end double talk should suspend adaptation.

    Compare the two frame energies.  Double talk is detected when the
    microphone energy is positive and::

        err_energy > threshold * mic_energy

    Thus a loud error alone is not enough: it must also be large relative to
    the microphone signal.  A non-positive ``threshold`` is treated as
    ``0.0``.  Silent or error-free frames return ``False``.  When this
    function returns ``True``, the caller must skip the NLMS weight update
    for the frame.
    """
    mic_energy = 0.0
    for sample in mic:
        mic_energy += sample * sample
    if mic_energy <= 0.0:
        return False

    if threshold <= 0.0:
        threshold = 0.0

    err_energy = 0.0
    for sample in err:
        err_energy += sample * sample

    return err_energy > threshold * mic_energy


def validate_params(length: int, mu: float) -> list[str]:
    """Return human-readable problems with filter parameters.

    A valid configuration has a positive integer ``length`` and a step size
    ``mu`` in ``[0.0, 2.0)``; zero freezes adaptation, while negative values
    or values at or above 2 are invalid.  Return an empty list when all
    parameters are valid.  Each problem string mentions either ``"length"``
    or ``"mu"`` so callers can present the issue directly.
    """
    problems: list[str] = []

    if (
        not isinstance(length, int)
        or isinstance(length, bool)
        or length <= 0
    ):
        problems.append(
            f"length must be a positive integer, got {length!r}"
        )

    if (
        not isinstance(mu, (int, float))
        or isinstance(mu, bool)
        or not math.isfinite(mu)
        or mu < 0.0
        or mu >= 2.0
    ):
        problems.append(
            f"mu must be a finite number in [0.0, 2.0), got {mu!r}"
        )

    return problems


def _is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def cancel_batch(pairs: list[tuple[list[float], list[float]]], config: dict) -> dict:
    """Cancel echo independently for every channel in a batch.

    ``pairs`` is a sequence of ``(mic, ref)`` pairs.  Within a pair the two
    signals have equal length.  ``config`` supplies the positive filter
    ``"length"`` and NLMS step ``"mu"``; optional keys are ``"max_delay"``
    (default ``0``) and ``"double_talk_threshold"`` (default ``0.5``).

    All channels are validated before any processing.  A malformed pair,
    mismatched signal lengths, or invalid config raises ``AecError`` and the
    call is atomic: it must yield no partial result and no channel may
    observe another channel's state.  Each channel starts with its own
    zero-initialized weight vector, estimates its own delay, aligns its own
    reference, and runs its own NLMS updates.

    Return a dictionary with one entry per input pair, in the original
    order::

        {
            "delays": [int, ...],
            "residuals": [list[float], ...],
            "weights": [list[float], ...],
            "erle_db": [float, ...],
        }
    """
    # ---- Configuration validation --------------------------------------
    if not isinstance(config, dict):
        raise AecError("config must be a dictionary")
    if "length" not in config or "mu" not in config:
        raise AecError("config must provide both 'length' and 'mu'")

    length = config["length"]
    mu = config["mu"]
    param_problems = validate_params(length, mu)
    if param_problems:
        raise AecError("invalid parameters: " + "; ".join(param_problems))

    max_delay = config.get("max_delay", 0)
    if (
        not isinstance(max_delay, int)
        or isinstance(max_delay, bool)
        or max_delay < 0
    ):
        raise AecError(
            f"max_delay must be a non-negative integer, got {max_delay!r}"
        )

    threshold = config.get("double_talk_threshold", 0.5)
    if not _is_finite_number(threshold):
        raise AecError(
            "double_talk_threshold must be a finite number, "
            f"got {threshold!r}"
        )

    # ---- Whole-batch validation before any processing ------------------
    if not isinstance(pairs, (list, tuple)):
        raise AecError("pairs must be a sequence of (mic, ref) pairs")

    checked_pairs: list[tuple[list[float], list[float]]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise AecError(
                f"pair {index} must contain exactly two signals"
            )
        mic, ref = pair
        if not isinstance(mic, list) or not isinstance(ref, list):
            raise AecError(
                f"pair {index}: mic and ref must both be lists"
            )
        if len(mic) != len(ref):
            raise AecError(
                f"pair {index}: mic and ref lengths must match "
                f"({len(mic)} != {len(ref)})"
            )
        if len(mic) == 0:
            raise AecError(f"pair {index}: signals must not be empty")
        if not all(_is_finite_number(v) for v in mic) or not all(
            _is_finite_number(v) for v in ref
        ):
            raise AecError(
                f"pair {index}: all samples must be finite numbers"
            )
        # Reuse the delay estimator's own validity rule (max_delay out of
        # range, etc.) so a bad batch is rejected up front.
        if delay_estimate(mic, ref, max_delay) < 0:
            raise AecError(
                f"pair {index}: max_delay {max_delay} is out of range for "
                f"signal length {len(mic)}"
            )
        checked_pairs.append((mic, ref))

    # ---- Per-channel processing; state is never shared -----------------
    delays: list[int] = []
    residuals: list[list[float]] = []
    all_weights: list[list[float]] = []
    erle_db: list[float] = []

    for mic, ref in checked_pairs:
        # Every channel gets its own delay estimate, aligned reference and
        # zero-initialized weight vector.
        delay = delay_estimate(mic, ref, max_delay)
        aligned = align_ref(ref, delay)
        n_samples = len(mic)

        # Build the tapped reference window for each sample once.  Samples
        # before time zero are implicitly zero.
        windows: list[list[float]] = []
        for t in range(n_samples):
            window = [0.0] * length
            for j in range(length):
                idx = t - (length - 1) + j
                if idx >= 0:
                    window[j] = float(aligned[idx])
            windows.append(window)

        # Online NLMS pass for this channel alone.  The double-talk
        # decision gates every individual update.  A fresh filter starts
        # with zero weights, so its residual equals the microphone and a
        # naive energy comparison would flag every cold-start sample as
        # double talk, permanently freezing adaptation even during
        # far-end-only speech.  The first ``guard_window`` samples
        # therefore always adapt (the convergence bootstrap); once the
        # filter is tracking, short trailing windows of microphone and
        # residual are compared so genuine near-end speech freezes the
        # weights while a residual that merely mirrors the microphone does
        # not.
        weights = [0.0] * length
        channel_residual: list[float] = []
        # Window long enough to hold the full filter tap depth.
        guard_window = max(length, 1)
        for t in range(n_samples):
            window = windows[t]
            echo_hat = estimate_echo(weights, window)
            error = float(mic[t]) - echo_hat
            channel_residual.append(error)

            if t < guard_window or not double_talk_guard(
                mic[t - guard_window + 1 : t + 1],
                channel_residual[t - guard_window + 1 : t + 1],
                threshold,
            ):
                # Far-end single talk (or silence): adapt along the
                # negative gradient with energy-normalized step size.
                weights = nlms_step(weights, window, error, mu)
            # Double talk after convergence: keep the frozen weights and
            # skip the update.

        delays.append(delay)
        residuals.append(channel_residual)
        all_weights.append(weights)
        erle_db.append(erle(mic, channel_residual))

    return {
        "delays": delays,
        "residuals": residuals,
        "weights": all_weights,
        "erle_db": erle_db,
    }
