"""Generate labeled synthetic tracking trials for model training."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .simulator import ModelParameters, simulate_response

def random_target(rng: np.random.Generator, samples: int, sampling_rate: float,
                  min_dwell_seconds: float = 0.3,
                  max_dwell_seconds: float = 1.5) -> np.ndarray:
    """Create a staircase target with randomized levels and dwell durations."""
    if samples < 2 or sampling_rate <= 0:
        raise ValueError("samples must be at least 2 and sampling_rate must be positive")
    if min_dwell_seconds <= 0 or max_dwell_seconds < min_dwell_seconds:
        raise ValueError("dwell limits must be positive and max_dwell_seconds >= min_dwell_seconds")
    min_dwell = max(1, int(round(min_dwell_seconds * sampling_rate)))
    max_dwell = max(min_dwell, int(round(max_dwell_seconds * sampling_rate)))
    target = np.empty(samples, dtype=np.float64)
    start = 0
    while start < samples:
        dwell = int(rng.integers(min_dwell, max_dwell + 1))
        end = min(samples, start + dwell)
        target[start:end] = rng.uniform(-1.0, 1.0)
        start = end
    return target

def generate_dataset(count: int = 10000, duration: float = 8.0,
                     sampling_rate: float = 100.0, seed: int = 42,
                     noise_std: float = 0.01) -> dict[str, np.ndarray]:
    """Generate input/output trials and known model parameter labels."""
    if count < 1 or duration <= 0 or sampling_rate <= 0 or noise_std < 0:
        raise ValueError("count/duration/sampling_rate must be positive and noise_std nonnegative")
    rng = np.random.default_rng(seed)
    samples = int(round(duration * sampling_rate))
    if samples < 2:
        raise ValueError("duration must contain at least two samples")
    inputs = np.empty((count, samples), dtype=np.float32)
    outputs = np.empty_like(inputs)
    labels = np.empty((count, 4), dtype=np.float32)
    for i in range(count):
        params = ModelParameters(
            gain=float(rng.uniform(0.6, 1.4)),
            natural_frequency=float(np.exp(rng.uniform(np.log(0.5), np.log(12.0)))),
            damping_ratio=float(np.exp(rng.uniform(np.log(0.25), np.log(2.0)))),
            delay=float(rng.uniform(0.0, 0.5)),
        )
        target = random_target(rng, samples, sampling_rate)
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
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic THRUST-synapse data.")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--duration", type=float, default=8.0, help="Trial duration in seconds.")
    parser.add_argument("--sampling-rate", type=float, default=100.0, help="Samples per second.")
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--min-dwell", type=float, default=0.3, help="Minimum target hold time in seconds.")
    parser.add_argument("--max-dwell", type=float, default=1.5, help="Maximum target hold time in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/train.npz"))
    args = parser.parse_args()
    data = generate_dataset(args.count, args.duration, args.sampling_rate, args.seed, args.noise_std)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(f"Saved {args.count} trials ({data['inputs'].shape[1]} samples each) to {args.output}")

if __name__ == "__main__":
    main()
