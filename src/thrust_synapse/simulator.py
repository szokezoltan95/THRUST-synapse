"""Simulation of a second-order transfer function with transport delay."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import cont2discrete, tf2ss

@dataclass(frozen=True)
class ModelParameters:
    """Second-order response with optional real zero and transport delay.

    G(s) = K*wn^2*(Tz*s + 1)/(s^2 + 2*zeta*wn*s + wn^2) * exp(-tau*s).
    A zero_time of zero preserves the original second-order model.
    """
    gain: float
    natural_frequency: float
    damping_ratio: float
    delay: float
    zero_time: float = 0.0

    def validate(self) -> None:
        if self.natural_frequency <= 0:
            raise ValueError("natural_frequency must be positive")
        if self.damping_ratio <= 0:
            raise ValueError("damping_ratio must be positive")
        if self.delay < 0:
            raise ValueError("delay must be nonnegative")
        if self.zero_time < 0:
            raise ValueError("zero_time must be nonnegative")

def delay_signal(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """Apply a nonnegative delay, using linear interpolation for fractional samples."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if delay_samples < 0:
        raise ValueError("delay_samples must be nonnegative")
    positions = np.arange(values.size, dtype=np.float64) - delay_samples
    return np.interp(positions, np.arange(values.size), values, left=values[0], right=values[-1])

def simulate_response(target: np.ndarray, parameters: ModelParameters,
                      sampling_rate: float = 100.0,
                      initial_output: float | None = None) -> np.ndarray:
    """Simulate sampled target using a zero-order hold.

    The system starts at rest relative to the first target value. By default,
    its output starts at gain * target[0]. An optional initial output offset
    decays according to the system dynamics.
    """
    u = np.asarray(target, dtype=np.float64)
    if u.ndim != 1 or u.size == 0:
        raise ValueError("target must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(u)):
        raise ValueError("target values must be finite")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    parameters.validate()
    delayed = delay_signal(u, parameters.delay * sampling_rate)
    wn, zeta = parameters.natural_frequency, parameters.damping_ratio
    scale = parameters.gain * wn * wn
    num = [scale * parameters.zero_time, scale] if parameters.zero_time > 0 else [scale]
    den = [1.0, 2.0 * zeta * wn, wn * wn]
    a, b, c, d, _ = cont2discrete(tf2ss(num, den), 1.0 / sampling_rate, method="zoh")
    state = np.zeros(a.shape[0], dtype=np.float64)
    result = np.empty_like(delayed)
    baseline = float(delayed[0])
    offset = parameters.gain * baseline if initial_output is None else float(initial_output)
    for i, value in enumerate(delayed):
        state = a @ state + b[:, 0] * (value - baseline)
        result[i] = (c @ state).item() + offset
    return result
