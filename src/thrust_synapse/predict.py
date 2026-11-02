"""Apply a trained estimator to one target/response pair in a THRUST TSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .simulator import ModelParameters, simulate_response
from .train import ParameterRegressor


def read_tsv(path: Path, input_column: str, output_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load timestamps and selected numeric columns from a tab-separated export."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {input_column, output_column, "TIME"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"TSV must contain columns: {', '.join(sorted(required))}")
        rows = list(reader)
    try:
        time_s = np.asarray([float(row["TIME"]) for row in rows], dtype=np.float64)
        target = np.asarray([float(row[input_column]) for row in rows], dtype=np.float64)
        response = np.asarray([float(row[output_column]) for row in rows], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Selected TSV columns must contain only numeric values") from exc
    valid = np.isfinite(time_s) & np.isfinite(target) & np.isfinite(response)
    time_s, target, response = time_s[valid], target[valid], response[valid]
    if len(time_s) < 2:
        raise ValueError("TSV contains fewer than two valid samples")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("TIME must be strictly increasing")
    return time_s, target, response


def predict_tsv(model_path: Path, tsv_path: Path, input_column: str, output_column: str,
                output_prefix: Path, scale: float = 900.0) -> dict[str, float]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ParameterRegressor()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    times, raw_target, raw_response = read_tsv(tsv_path, input_column, output_column)
    sampling_rate = float(checkpoint["sampling_rate_hz"])
    uniform_time = np.arange(times[0], times[-1], 1.0 / sampling_rate)
    if uniform_time.size < 2:
        raise ValueError("TSV duration is too short for the checkpoint sampling rate")
    target = np.interp(uniform_time, times, raw_target) / scale
    response = np.interp(uniform_time, times, raw_response) / scale
    features = np.stack((target, response), axis=0).astype(np.float32)[None, :, :]
    with torch.no_grad():
        normalized = model(torch.from_numpy(features)).numpy()[0]

    mean = np.asarray(checkpoint["target_mean"], dtype=np.float64)
    std = np.asarray(checkpoint["target_std"], dtype=np.float64)
    transformed = normalized * std + mean
    parameters = transformed.copy()
    parameters[:3] = np.exp(parameters[:3])
    gain, natural_frequency, damping_ratio, delay = (float(v) for v in parameters)
    safe_delay = max(0.0, delay)
    model_response = simulate_response(
        target - target[0],
        ModelParameters(gain, natural_frequency, damping_ratio, safe_delay),
        sampling_rate=sampling_rate,
        initial_output=float(response[0]),
    )
    rmse = float(np.sqrt(np.mean((model_response - response) ** 2)))

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "gain": gain,
        "natural_frequency_rad_s": natural_frequency,
        "damping_ratio": damping_ratio,
        "delay_s": delay,
        "response_rmse_normalized": rmse,
        "duration_s": float(uniform_time[-1] - uniform_time[0]),
        "sampling_rate_hz": sampling_rate,
    }
    json_path = output_prefix.with_suffix(".json")
    plot_path = output_prefix.with_name(output_prefix.name + "_response.png")
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(uniform_time - uniform_time[0], target, color="0.55", linewidth=0.9)
    axes[0].set_ylabel(f"Target ({input_column}) / {scale:g}")
    axes[1].plot(uniform_time - uniform_time[0], response, label="Measured response", linewidth=0.8)
    axes[1].plot(uniform_time - uniform_time[0], model_response, label="Estimated model", linewidth=1.1)
    axes[1].set_ylabel(f"Response ({output_column}) / {scale:g}")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"{input_column} → {output_column}: K={gain:.3f}, wn={natural_frequency:.2f} rad/s, "
        f"zeta={damping_ratio:.3f}, delay={delay:.3f} s, RMSE={rmse:.4f}"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    for key, value in metrics.items():
        print(f"{key}: {value:.6g}")
    print(f"Saved metrics to {json_path}")
    print(f"Saved response plot to {plot_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate transfer-function parameters from a THRUST TSV.")
    parser.add_argument("--model", type=Path, required=True, help="Trained .pt checkpoint.")
    parser.add_argument("--tsv", type=Path, required=True, help="THRUST tab-separated export.")
    parser.add_argument("--input-column", required=True, help="Target/request column, e.g. LXRQ.")
    parser.add_argument("--output-column", required=True, help="Measured stick column, e.g. LX.")
    parser.add_argument("--output-prefix", type=Path, default=Path("results/real_scope"))
    parser.add_argument("--scale", type=float, default=900.0,
                        help="Divide raw target and response by this value before inference.")
    args = parser.parse_args()
    predict_tsv(args.model, args.tsv, args.input_column, args.output_column,
                args.output_prefix, args.scale)


if __name__ == "__main__":
    main()
