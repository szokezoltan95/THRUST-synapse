"""Generate labeled synthetic tracking trials for model training."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .simulator import ModelParameters, simulate_response

def random_target(rng: np.random.Generator, samples: int, sampling_rate: float,
                  dwell_seconds: float = 3.0) -> np.ndarray:
    """Create a staircase target that changes at a fixed dwell interval."""
    if samples < 2 or sampling_rate <= 0 or dwell_seconds <= 0:
        raise ValueError("samples, sampling_rate, and dwell_seconds must be positive")
    dwell_samples = max(1, int(round(dwell_seconds * sampling_rate)))
    target = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, dwell_samples):
        end = min(samples, start + dwell_samples)
        target[start:end] = rng.uniform(-1.0, 1.0)
    return target

def generate_dataset(count: int = 10000, duration: float = 8.0,
                     sampling_rate: float = 100.0, seed: int = 42,
                     noise_std: float = 0.01, dwell_seconds: float = 3.0,
                     profile: str = "scope_x") -> dict[str, np.ndarray]:
    """Generate input/output trials and known model parameter labels."""
    if count < 1 or duration <= 0 or sampling_rate <= 0 or noise_std < 0:
        raise ValueError("count/duration/sampling_rate must be positive and noise_std nonnegative")
    if profile not in {"scope_x", "broad"}:
        raise ValueError("profile must be 'scope_x' or 'broad'")
    rng = np.random.default_rng(seed)
    samples = int(round(duration * sampling_rate))
    if samples < 2:
        raise ValueError("duration must contain at least two samples")
    inputs = np.empty((count, samples), dtype=np.float32)
    outputs = np.empty_like(inputs)
    labels = np.empty((count, 4), dtype=np.float32)
    for i in range(count):
        if profile == "scope_x":
            # Approximate fit to the two X-axis channels in the supplied
            # measurement: wn 7.5-11.4 rad/s, zeta 0.89-0.97,
            # gain 0.98-0.99, and delay 0.24-0.28 s.
            params = ModelParameters(
                gain=float(np.clip(rng.normal(0.986, 0.07), 0.75, 1.20)),
                natural_frequency=float(np.clip(
                    rng.lognormal(np.log(9.3), 0.20), 5.0, 15.0
                )),
                damping_ratio=float(np.clip(rng.normal(0.93, 0.12), 0.60, 1.30)),
                delay=float(np.clip(rng.normal(0.258, 0.06), 0.08, 0.45)),
            )
        else:
            params = ModelParameters(
                gain=float(rng.uniform(0.6, 1.4)),
                natural_frequency=float(np.exp(rng.uniform(np.log(0.5), np.log(12.0)))),
                damping_ratio=float(np.exp(rng.uniform(np.log(0.25), np.log(2.0)))),
                delay=float(rng.uniform(0.0, 0.5)),
            )
        target = random_target(rng, samples, sampling_rate, dwell_seconds)
        response = simulate_response(target, params, sampling_rate)
        if noise_std:
            response += rng.normal(0.0, noise_std, size=samples)
        inputs[i], outputs[i] = target, response
        labels[i] = (params.gain, params.natural_frequency,
                     params.damping_ratio, params.delay)
    return {
        "inputs": inputs, "outputs": outputs, "parameters": labels,
        "parameter_names": np.asarray(["gain", "natural_frequency_rad_s",
                                       "damping_ratio", "delay_s"]),
        "sampling_rate_hz": np.asarray(sampling_rate, dtype=np.float32),
        "seed": np.asarray(seed, dtype=np.int64),
        "parameter_profile": np.asarray(profile),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic THRUST-synapse data.")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--duration", type=float, default=8.0, help="Trial duration in seconds.")
    parser.add_argument("--sampling-rate", type=float, default=100.0, help="Samples per second.")
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--profile", choices=["scope_x", "broad"], default="scope_x",
                        help="Parameter distribution: fitted X-axis range or original broad range.")
    parser.add_argument("--dwell-seconds", type=float, default=3.0,
                        help="Fixed time between target changes in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/train.npz"))
    args = parser.parse_args()
    data = generate_dataset(args.count, args.duration, args.sampling_rate, args.seed, args.noise_std, args.dwell_seconds, args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(f"Saved {args.count} trials ({data['inputs'].shape[1]} samples each) to {args.output}")

if __name__ == "__main__":
    main()
