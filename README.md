# THRUST-synapse

**Local neural identification of human tracking dynamics.**

THRUST-synapse is an experimental research component of the **THRUST** ecosystem, developed in the context of UAV pilot performance research at the Faculty of Aeronautics, Technical University of Košice.

The project explores neural networks that estimate interpretable transfer-function parameters directly from a target trajectory and the corresponding measured control-stick response.

> **Status: early development.** The repository contains a second-order delayed-system simulator and a reproducible synthetic dataset generator. Neural-network training, real-data workflows, and THRUST integration are not implemented yet.

## Initial model

The baseline is a second-order system with transport delay:

$$
G(s) = K \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2} e^{-\tau s}
$$

Parameters are gain $K$, natural angular frequency $\omega_n$ in rad/s, damping ratio $\zeta$, and delay $\tau$ in seconds. Damping values below and above one allow both underdamped complex poles and non-oscillatory real poles.

The simulator uses a zero-order hold at the selected sampling rate and linearly interpolates fractional-sample delays. Its initial input/output definition is target position to measured stick position.

## Quick start

Install the package in editable mode:

```bash
python -m pip install -e .
```

Generate 10,000 eight-second trials sampled at 100 Hz:

```bash
thrust-synapse-generate --count 10000 --duration 8 --sampling-rate 100 --seed 42 --output data/synthetic/train.npz
```

The generator is also callable from Python:

```python
from thrust_synapse.dataset import generate_dataset

dataset = generate_dataset(count=1000, duration=8.0, sampling_rate=100.0, seed=42)
```

Each compressed NPZ dataset contains:

| Array | Shape | Description |
| --- | --- | --- |
| `inputs` | `(trials, samples)` | Random piecewise-constant target trajectories |
| `outputs` | `(trials, samples)` | Simulated responses with additive Gaussian measurement noise |
| `parameters` | `(trials, 4)` | Ground-truth labels in the order gain, natural frequency, damping ratio, delay |
| `parameter_names` | `(4,)` | Names and units for the label columns |
| `sampling_rate_hz` | scalar | Sampling rate |
| `seed` | scalar | Random seed for reproducibility |

Default parameter ranges are gain 0.6–1.4, natural frequency 0.5–12 rad/s (log-uniform), damping ratio 0.25–2 (log-uniform), and delay 0–0.5 seconds. The default output noise standard deviation is 0.01 target units. These are initial engineering ranges for experimentation, not ranges established from participant data.

Use different seeds for separate dataset splits. Do not place augmented copies of the same base trial in different splits. The generator currently creates piecewise-constant targets; richer target patterns and experimentally calibrated noise will be added as the real measurement workflow is defined.

## Development status

- [x] Implement the delayed second-order simulator.
- [x] Generate labeled synthetic trials with a fixed seed.
- [ ] Add validation and configurable parameter bounds.
- [ ] Add synthetic data inspection and dataset split utilities.
- [ ] Train a baseline neural parameter estimator.
- [ ] Add response-based training on real measurements.
- [ ] Benchmark recovery, robustness, and runtime.
- [ ] Explore integration with THRUST-compute.

Synthetic parameter recovery alone will not establish that the model describes human behavior. Real measurements are needed to assess response prediction and parameter consistency. Different parameter combinations may produce similar responses, so fit quality does not guarantee unique or physiologically exact parameters.

Experimental measurements and participant data are not intended to be committed to this repository. Python and PyTorch are the proposed foundation for training and inference; PyTorch will be added when the training component is implemented.
