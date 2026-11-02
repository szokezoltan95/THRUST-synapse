"""Train and evaluate a baseline neural parameter estimator."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .simulator import ModelParameters, simulate_response


class ParameterRegressor(nn.Module):
    """Small 1-D convolutional model for target/response trajectory pairs."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(128 * 16, 128),
            nn.GELU(),
            nn.Linear(128, 4),
        )

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        return self.features(trajectories)


def _transform_parameters(values: np.ndarray) -> np.ndarray:
    """Log-transform positive-scale parameters; keep delay linear."""
    result = values.astype(np.float64, copy=True)
    result[:, :3] = np.log(np.maximum(result[:, :3], 1e-8))
    return result


def _inverse_parameters(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    result[:, :3] = np.exp(result[:, :3])
    return result


def _response_rmse(inputs: np.ndarray, outputs: np.ndarray, parameters: np.ndarray,
                   sampling_rate: float, max_trials: int = 512) -> float:
    errors = []
    for target, observed, row in zip(inputs[:max_trials], outputs[:max_trials], parameters[:max_trials]):
        predicted = simulate_response(
            target,
            ModelParameters(float(row[0]), float(row[1]), float(row[2]), float(max(0.0, row[3]))),
            sampling_rate,
        )
        errors.append(np.mean((predicted - observed) ** 2))
    return float(np.sqrt(np.mean(errors)))


def _save_diagnostics(output_path: Path, inputs: np.ndarray, outputs: np.ndarray,
                      actual: np.ndarray, predicted: np.ndarray, indices: np.ndarray,
                      sampling_rate: float, seed: int) -> tuple[Path, Path, Path]:
    """Save parameter comparisons, response examples, and raw test predictions."""
    names = ["Gain K", "Natural frequency (rad/s)", "Damping ratio", "Delay (s)"]
    parameter_plot = output_path.with_name(output_path.stem + "_parameters.png")
    response_plot = output_path.with_name(output_path.stem + "_responses.png")
    prediction_file = output_path.with_name(output_path.stem + "_test_predictions.npz")
    np.savez_compressed(
        prediction_file,
        actual_parameters=actual,
        predicted_parameters=predicted,
        test_indices=indices,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for column, ax in enumerate(axes.flat):
        truth, estimate = actual[:, column], predicted[:, column]
        low = float(min(truth.min(), estimate.min()))
        high = float(max(truth.max(), estimate.max()))
        pad = max((high - low) * 0.05, 1e-6)
        ax.scatter(truth, estimate, s=12, alpha=0.45)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1)
        ax.set_title(names[column])
        ax.set_xlabel("True")
        ax.set_ylabel("Estimated")
        ax.grid(alpha=0.25)
    fig.suptitle("Held-out synthetic test set: parameter estimates")
    fig.tight_layout()
    fig.savefig(parameter_plot, dpi=160)
    plt.close(fig)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(indices), size=min(6, len(indices)), replace=False)
    fig, axes = plt.subplots(len(chosen), 1, figsize=(11, max(2.5 * len(chosen), 5)), squeeze=False)
    time_axis = np.arange(inputs.shape[1]) / sampling_rate
    for ax, test_pos in zip(axes.flat, chosen):
        row = predicted[test_pos]
        model_response = simulate_response(
            inputs[test_pos],
            ModelParameters(float(row[0]), float(row[1]), float(row[2]), float(max(0.0, row[3]))),
            sampling_rate,
        )
        ax.plot(time_axis, inputs[test_pos], color="0.65", linewidth=1, label="Target")
        ax.plot(time_axis, outputs[test_pos], color="tab:blue", alpha=0.7, linewidth=1, label="Observed")
        ax.plot(time_axis, model_response, color="tab:orange", linewidth=1.4, label="Estimated model")
        ax.set_ylabel("Position")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(loc="upper right", ncols=3, fontsize=8)
    axes.flat[-1].set_xlabel("Time (s)")
    fig.suptitle("Held-out test response examples")
    fig.tight_layout()
    fig.savefig(response_plot, dpi=160)
    plt.close(fig)
    return parameter_plot, response_plot, prediction_file


def train(data_path: Path, output_path: Path, epochs: int = 100,
          batch_size: int = 128, seed: int = 123) -> dict[str, float]:
    try:
        archive = np.load(data_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read dataset: {data_path}") from exc
    required = {"inputs", "outputs", "parameters", "sampling_rate_hz"}
    if not required.issubset(archive.files):
        raise ValueError(f"Dataset is missing arrays: {sorted(required - set(archive.files))}")
    x_in = np.asarray(archive["inputs"], dtype=np.float32)
    x_out = np.asarray(archive["outputs"], dtype=np.float32)
    labels = np.asarray(archive["parameters"], dtype=np.float32)
    sampling_rate = float(np.asarray(archive["sampling_rate_hz"]).item())
    if x_in.ndim != 2 or x_out.shape != x_in.shape or labels.shape != (len(x_in), 4):
        raise ValueError("Expected inputs/outputs shaped (trials, samples) and parameters shaped (trials, 4)")
    if len(x_in) < 10:
        raise ValueError("At least 10 trials are needed for train/validation/test splits")
    if not all(np.isfinite(a).all() for a in (x_in, x_out, labels)):
        raise ValueError("Dataset contains NaN or infinite values")
    if np.any(labels[:, :3] <= 0) or np.any(labels[:, 3] < 0):
        raise ValueError("Expected positive gain/frequency/damping and nonnegative delay")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_in))
    n_train = int(0.8 * len(order))
    n_val = int(0.1 * len(order))
    splits = order[:n_train], order[n_train:n_train+n_val], order[n_train+n_val:]
    train_idx, val_idx, test_idx = splits
    transformed = _transform_parameters(labels)
    mean = transformed[train_idx].mean(axis=0)
    std = transformed[train_idx].std(axis=0)
    std[std < 1e-8] = 1.0
    y_scaled = (transformed - mean) / std

    trajectories = np.stack((x_in, x_out), axis=1)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParameterRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_data = TensorDataset(torch.from_numpy(trajectories[train_idx]),
                               torch.from_numpy(y_scaled[train_idx].astype(np.float32)))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    x_val = torch.from_numpy(trajectories[val_idx]).to(device)
    y_val = torch.from_numpy(y_scaled[val_idx].astype(np.float32)).to(device)
    best_loss, best_state, patience = float("inf"), None, 15

    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(xb)
            train_sample_count += len(xb)
        train_loss = train_loss_sum / train_sample_count
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).item())
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 15
        else:
            patience -= 1
        status = "best" if improved else f"patience {patience}/15"
        elapsed = time.perf_counter() - epoch_started
        print(
            f"Epoch {epoch + 1:03d}/{epochs} | train loss {train_loss:.6f} "
            f"| val loss {val_loss:.6f} | {status} | {elapsed:.1f}s",
            flush=True,
        )
        if patience == 0:
            print("Early stopping: validation loss did not improve for 15 epochs.", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(torch.from_numpy(trajectories[test_idx]).to(device)).cpu().numpy()
    prediction = _inverse_parameters(prediction_scaled * std + mean)
    actual = labels[test_idx].astype(np.float64)
    parameter_mae = np.mean(np.abs(prediction - actual), axis=0)
    relative_mae = np.mean(np.abs(prediction - actual) / np.maximum(np.abs(actual), 1e-6), axis=0)
    response_rmse = _response_rmse(x_in[test_idx], x_out[test_idx], prediction, sampling_rate)
    reference_rmse = _response_rmse(x_in[test_idx], x_out[test_idx], actual, sampling_rate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_plot, response_plot, prediction_file = _save_diagnostics(
        output_path, x_in[test_idx], x_out[test_idx], actual, prediction, test_idx, sampling_rate, seed
    )
    torch.save({
        "state_dict": model.state_dict(),
        "target_mean": mean,
        "target_std": std,
        "sample_count": int(x_in.shape[1]),
        "sampling_rate_hz": sampling_rate,
        "parameter_names": ["gain", "natural_frequency_rad_s", "damping_ratio", "delay_s"],
    }, output_path)
    metrics = {
        "best_validation_loss": best_loss,
        "test_gain_mae": float(parameter_mae[0]),
        "test_natural_frequency_mae": float(parameter_mae[1]),
        "test_damping_ratio_mae": float(parameter_mae[2]),
        "test_delay_mae_s": float(parameter_mae[3]),
        "test_mean_relative_parameter_error": float(relative_mae.mean()),
        "test_response_rmse": response_rmse,
        "test_reference_rmse_true_parameters": reference_rmse,
        "train_trials": float(len(train_idx)),
        "validation_trials": float(len(val_idx)),
        "test_trials": float(len(test_idx)),
    }
    print(f"Device: {device}; best validation loss: {best_loss:.6f}")
    for name, value in metrics.items():
        print(f"{name}: {value:.6g}")
    print(f"Saved model to {output_path}")
    print(f"Saved parameter plot to {parameter_plot}")
    print(f"Saved response plot to {response_plot}")
    print(f"Saved test predictions to {prediction_file}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline THRUST-synapse parameter estimator.")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/train.npz"))
    parser.add_argument("--output", type=Path, default=Path("models/baseline.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123, help="Seed for the train/validation/test split.")
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
