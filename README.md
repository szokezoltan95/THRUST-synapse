# THRUST-synapse

**Local neural identification of human tracking dynamics.**

THRUST-synapse is an experimental research component of the **THRUST** ecosystem, developed in the context of UAV pilot performance research at the Faculty of Aeronautics, Technical University of Košice.

The project explores neural networks that estimate interpretable transfer-function parameters directly from a target trajectory and the corresponding measured control-stick response.

> **Status: project initialization.** This repository currently contains the project outline. Dataset generation, training, inference, and integration are planned; no trained model or runnable application is available yet.

## Purpose

The intended workflow is to train a network once and then estimate model parameters from new measurements using a forward pass, without a separate iterative parameter search for each measurement.

The output remains a classical transfer function, allowing simulated responses, parameter comparisons, and validation alongside conventional identification methods. Speed and accuracy improvements must be established experimentally.

## Initial model

The proposed baseline is a second-order system with transport delay:

$$
G(s) = K \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2} e^{-\tau s}
$$

| Parameter | Meaning |
| --- | --- |
| $K$ | Steady-state gain |
| $\omega_n$ | Natural angular frequency, in rad/s |
| $\zeta$ | Damping ratio |
| $\tau$ | Transport delay, in seconds |

Positive natural frequency and damping provide a stable dynamic component. Allowing damping below and above one supports both complex-conjugate and real poles. Delay is constrained to be nonnegative.

The initial input/output definition is **target position → measured stick position**. This describes overall tracking behavior; an error-to-stick operator model would require a separately defined identification setup.

## Planned learning workflow

1. **Synthetic pretraining** — simulate known parameter combinations, varied target trajectories, initial conditions, and realistic measurement disturbances.
2. **Learning from real measurements** — use a differentiable simulator to reconstruct measured responses from predicted parameters. Training minimizes response mismatch without requiring ground-truth parameter labels.
3. **Independent validation** — evaluate parameter recovery on synthetic tests and response reconstruction and prediction on held-out real trials.
4. **Local inference** — estimate parameters and report response agreement for new measurements.

Synthetic variants derived from the same underlying example must remain in the same dataset split. Real-data splits should account for participants and sessions to avoid leakage and distinguish within-participant from unseen-participant performance.

## THRUST data and integration

The intended data source is THRUST tracking measurements, initially focusing on SCoPE:

- Nominal sampling rate: **100 Hz**.
- Standard stick channels: **LX, LY, RX, RY**.
- Corresponding target/request channels where recorded.
- Planned compatibility with THRUST exports and the existing analysis workflow.

Trial duration and preprocessing will be configurable. Longer windows can help capture settling and oscillations, but the usable window must reflect actual task boundaries.

| Component | Role |
| --- | --- |
| THRUST-measure | Measurement acquisition |
| THRUST-webdb | Measurement storage and access |
| THRUST-compute | Existing analysis and conventional model identification |
| **THRUST-synapse** | Experimental neural parameter estimation |

## Validation principles

A close fit does not establish unique or physiologically exact parameters. Different parameter combinations may produce similar responses, and human behavior may depart from the selected model structure.

Evaluation should therefore include:

- Parameter recovery on unseen synthetic examples.
- Response error on held-out real trials.
- Parameter consistency across comparable tasks.
- Comparison with conventional identification using the same data and model structure.
- Inference runtime and robustness to noise and model mismatch.

Experimental measurements and participant data are not intended to be committed to this repository.

## Initial roadmap

- [ ] Define parameter bounds, preprocessing, and dataset format.
- [ ] Implement and verify the delayed second-order simulator.
- [ ] Build a reproducible synthetic dataset generator.
- [ ] Train a baseline neural parameter estimator.
- [ ] Add differentiable response-based training on real data.
- [ ] Benchmark accuracy, robustness, and runtime.
- [ ] Export model parameters, validation metrics, and response plots.
- [ ] Explore integration with THRUST-compute.

## Planned implementation

Python and PyTorch are the proposed foundation for local training and inference. CPU execution is intended, with optional GPU acceleration for training. Dependencies and installation instructions will be added with the first implementation.
