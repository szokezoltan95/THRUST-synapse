# THRUST-synapse

**Local neural identification of human tracking dynamics.**

THRUST-synapse is an experimental research component of the **THRUST** ecosystem, developed in the context of UAV pilot performance research at the Faculty of Aeronautics, Technical University of Košice.

The project explores neural networks that estimate interpretable transfer-function parameters directly from a target trajectory and the corresponding measured control-stick response.

> **Status: early development.** The repository contains a delayed second-order simulator, a synthetic dataset generator, and an initial neural parameter estimator. Validation on real measurements and THRUST integration remain future work.

## Initial model

The baseline is a second-order system with transport delay:

$$
G(s) = K \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2} e^{-\tau s}
$$

Parameters are gain $K$, natural angular frequency $\omega_n$ in rad/s, damping ratio $\zeta$, and delay $\tau$ in seconds. Damping values below and above one allow both underdamped complex poles and non-oscillatory real poles.

The simulator uses a zero-order hold at the selected sampling rate and linearly interpolates fractional-sample delays. Its initial input/output definition is target position to measured stick position.

## Quick start

Install the package and training dependency:

```bash
python -m pip install -e ".[train]"
```

Generate 10,000 eight-second trials sampled at 100 Hz, with target levels held for 0.3–1.5 seconds:

```bash
thrust-synapse-generate --count 10000 --duration 8 --sampling-rate 100 --seed 42 --min-dwell 0.3 --max-dwell 1.5 --output data/synthetic/train.npz
```

Train the baseline estimator with an automatic 80/10/10 train/validation/test split:

```bash
thrust-synapse-train --data data/synthetic/train.npz --output models/baseline.pt --epochs 100 --seed 123
```

The split seed is independent of the dataset generation seed. Keep the test set for final evaluation; use validation performance for model selection. The training script reports parameter errors and response RMSE on the held-out test portion. It also saves parameter comparison and response example plots, plus an NPZ file with test predictions, beside the model. The true-parameter response RMSE provides a reference for the measurement-noise floor.

## Dataset format

Each compressed NPZ dataset contains:

| Array | Shape | Description |
| --- | --- | --- |
| `inputs` | `(trials, samples)` | Random piecewise-constant target trajectories |
| `outputs` | `(trials, samples)` | Simulated responses with additive Gaussian measurement noise |
| `parameters` | `(trials, 4)` | Ground-truth labels: gain, natural frequency, damping ratio, delay |
| `parameter_names` | `(4,)` | Names and units for the label columns |
| `sampling_rate_hz` | scalar | Sampling rate |
| `seed` | scalar | Random seed for reproducibility |

Default parameter ranges are gain 0.6–1.4, natural frequency 0.5–12 rad/s (log-uniform), damping ratio 0.25–2 (log-uniform), and delay 0–0.5 seconds. The default output noise standard deviation is 0.01 target units. These are initial engineering ranges for experimentation, not ranges established from participant data.

The generator creates piecewise-constant staircase targets, with configurable randomized dwell times (default 0.3–1.5 seconds). These are initial excitation settings; their usefulness should be checked against validation results.

## Development status

- [x] Implement the delayed second-order simulator.
- [x] Generate labeled synthetic trials with a fixed seed.
- [x] Add a baseline convolutional parameter estimator and held-out evaluation.
- [ ] Add configurable parameter bounds and dataset inspection utilities.
- [x] Use staircase targets with configurable dwell durations.
- [x] Save held-out parameter and response diagnostic plots.
- [ ] Add response-based training on real measurements.
- [ ] Benchmark recovery, robustness, and runtime.
- [ ] Explore integration with THRUST-compute.

Synthetic parameter recovery alone will not establish that the model describes human behavior. Real measurements are needed to assess response prediction and parameter consistency. Different parameter combinations may produce similar responses, so fit quality does not guarantee unique or physiologically exact parameters.

Experimental measurements and participant data are not intended to be committed to this repository. Training runs locally using CPU or CUDA when available. PyTorch is an optional dependency used by the training command.
