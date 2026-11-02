"""Response-based refinement of neural transfer-function estimates."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .simulator import ModelParameters, simulate_response


def _unpack(values: np.ndarray) -> tuple[ModelParameters, float]:
    """Decode log-scaled physical parameters and a constant response offset."""
    gain, natural_frequency, damping_ratio = np.exp(values[:3])
    return (
        ModelParameters(float(gain), float(natural_frequency), float(damping_ratio), float(values[3])),
        float(values[4]),
    )


def fit_average_step_response(
    time_axis: np.ndarray,
    step_responses: dict[str, np.ndarray],
    sampling_rate: float,
    initial: ModelParameters,
) -> tuple[ModelParameters, float]:
    """Refine parameters against event-aligned mean step responses.

    The fit is deliberately based on isolated, aligned real target changes. It
    searches multiple damping-ratio starts so a neural estimate trained mostly
    on heavily damped synthetic examples does not prevent finding visible
    underdamped oscillations.
    """
    traces = [
        values for values in step_responses.values()
        if values.ndim == 2 and len(values) > 0
    ]
    if not traces:
        raise ValueError("No isolated step responses are available for refinement")
    observed = np.concatenate(traces, axis=0)
    if observed.shape[1] != len(time_axis):
        raise ValueError("Step response lengths do not match the time axis")
    if len(observed) < 3:
        raise ValueError("At least three isolated steps are required for refinement")

    # Weight each time sample equally, while soft_l1 limits the effect of
    # isolated spikes and imperfect event alignment.
    measured_mean = np.mean(observed, axis=0)
    measured_spread = np.std(observed, axis=0)
    weights = 1.0 / np.sqrt(1.0 + (measured_spread / 0.05) ** 2)

    unit_step = np.ones(len(time_axis), dtype=np.float64)
    unit_step[0] = 0.0

    def residual(vector: np.ndarray) -> np.ndarray:
        parameters, offset = _unpack(vector)
        simulated = simulate_response(
            unit_step, parameters, sampling_rate, initial_output=offset
        )
        return (simulated - measured_mean) * weights

    lower = np.array([np.log(0.2), np.log(1.0), np.log(0.025), 0.0, -1.5])
    upper = np.array([np.log(2.5), np.log(35.0), np.log(2.0), 1.0, 1.5])
    base = np.array([
        np.log(np.clip(initial.gain, 0.2, 2.5)),
        np.log(np.clip(initial.natural_frequency, 1.0, 35.0)),
        np.log(np.clip(initial.damping_ratio, 0.025, 2.0)),
        np.clip(initial.delay, 0.0, 1.0),
        float(measured_mean[0]),
    ])
    damping_starts = sorted({
        float(np.clip(initial.damping_ratio, 0.03, 1.95)),
        0.08, 0.16, 0.30, 0.55, 0.90,
    })

    baseline_rmse = float(np.sqrt(np.mean(residual(base) ** 2)))
    best = (baseline_rmse, base.copy())
    improved = False
    for damping in damping_starts:
        start = base.copy()
        start[2] = np.log(damping)
        result = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.03,
            max_nfev=80,
            x_scale="jac",
        )
        rmse = float(np.sqrt(np.mean(residual(result.x) ** 2)))
        if rmse < best[0]:
            best = (rmse, result.x)
            improved = True

    if not improved:
        raise ValueError("Mean-step fit did not improve on the neural estimate")
    if best[1][0] >= upper[0] - 1e-3:
        raise ValueError("Refined gain hit its upper search limit; keeping the neural estimate")

    parameters, _ = _unpack(best[1])
    return parameters, best[0]
