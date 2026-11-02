"""Apply a trained estimator to one target/response pair in a THRUST TSV."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .simulator import ModelParameters, simulate_response
from .refine import fit_average_step_response
from .train import ParameterRegressor


def _save_figure(fig: plt.Figure, path: Path, dpi: int = 160) -> Path:
    """Save a plot, using a unique filename if Windows has the old one locked."""
    try:
        fig.savefig(path, dpi=dpi)
        return path
    except OSError as first_error:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        try:
            fig.savefig(fallback, dpi=dpi)
        except OSError as fallback_error:
            raise OSError(
                f"Could not save plot to {path} or fallback path {fallback}. "
                "Check that the output folder is writable."
            ) from fallback_error
        print(f"Could not overwrite {path}; saved a new copy to {fallback}", flush=True)
        return fallback


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


def extract_average_steps(target: np.ndarray, response: np.ndarray, sampling_rate: float,
                          window_seconds: float = 3.0, baseline_seconds: float = 0.2,
                          change_threshold: float = 0.05,
                          merge_seconds: float = 0.15) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Align isolated target-step events and normalize by each full step size.

    A single requested level change can span adjacent samples in a resampled
    log. Group threshold crossings close in time so a partial edge increment
    is not mistaken for the full target step.
    """
    window = int(round(window_seconds * sampling_rate))
    baseline = max(1, int(round(baseline_seconds * sampling_rate)))
    merge_gap = max(1, int(round(merge_seconds * sampling_rate)))
    raw_changes = np.flatnonzero(np.abs(np.diff(target)) >= change_threshold) + 1

    groups: list[list[int]] = []
    for index in raw_changes:
        if not groups or index - groups[-1][-1] > merge_gap:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))

    collected: dict[str, list[np.ndarray]] = {"increasing": [], "decreasing": []}
    for position, group in enumerate(groups):
        index = group[0]
        end_index = group[-1]
        next_change = groups[position + 1][0] if position + 1 < len(groups) else len(target)
        if index < baseline or index + window > len(target) or next_change < index + window:
            continue

        step = float(target[end_index] - target[index - 1])
        if abs(step) < change_threshold:
            continue
        y0 = float(np.median(response[index - baseline:index]))
        normalized_response = (response[index:index + window] - y0) / step
        direction = "increasing" if step > 0 else "decreasing"
        collected[direction].append(normalized_response)

    arrays = {
        direction: np.asarray(rows, dtype=np.float64)
        if rows else np.empty((0, window), dtype=np.float64)
        for direction, rows in collected.items()
    }
    return np.arange(window, dtype=np.float64) / sampling_rate, arrays

def save_average_response_plot(path: Path, time_axis: np.ndarray,
                               step_responses: dict[str, np.ndarray],
                               parameters: ModelParameters, sampling_rate: float,
                               refined_parameters: ModelParameters | None = None) -> dict[str, int]:
    """Plot event-aligned measured step responses and the estimated model step response."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    counts: dict[str, int] = {}
    unit_step = np.ones(time_axis.size, dtype=np.float64)
    unit_step[0] = 0.0
    model_step = simulate_response(unit_step, parameters, sampling_rate, initial_output=0.0)
    refined_step = (simulate_response(unit_step, refined_parameters, sampling_rate, initial_output=0.0)
                    if refined_parameters is not None else None)

    for ax, direction in zip(axes, ("increasing", "decreasing")):
        rows = step_responses[direction]
        counts[direction] = int(len(rows))
        ax.set_title(f"{direction.capitalize()} target steps (n={len(rows)})")
        if len(rows):
            mean = rows.mean(axis=0)
            spread = rows.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros_like(mean)
            for row in rows:
                ax.plot(time_axis, row, color="tab:blue", alpha=0.12, linewidth=0.8)
            ax.fill_between(time_axis, mean - spread, mean + spread,
                            color="tab:blue", alpha=0.18, label="Mean ± 1 SD")
            ax.plot(time_axis, mean, color="tab:blue", linewidth=2, label="Measured mean")
            ax.plot(time_axis, model_step, color="tab:orange", linewidth=1.4,
                    label="Neural estimate")
            if refined_step is not None:
                ax.plot(time_axis, refined_step, color="tab:green", linewidth=1.8,
                        label=f"Refined fit (Tz={refined_parameters.zero_time:.3f} s)")
        else:
            ax.text(0.5, 0.5, "No isolated steps available", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_ylabel("Normalized response / step")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time after target change (s)")
    fig.suptitle("Event-aligned average response")
    fig.tight_layout()
    _save_figure(fig, path, dpi=160)
    plt.close(fig)
    return counts


def predict_tsv(model_path: Path, tsv_path: Path, input_column: str, output_column: str,
                output_prefix: Path, scale: float = 900.0,
                refine: bool = True) -> dict[str, float | int | bool | None]:
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
    neural_parameters = ModelParameters(gain, natural_frequency, damping_ratio, safe_delay)
    neural_response = simulate_response(
        target - target[0], neural_parameters, sampling_rate=sampling_rate,
        initial_output=float(response[0]),
    )
    neural_rmse = float(np.sqrt(np.mean((neural_response - response) ** 2)))
    average_time, average_steps = extract_average_steps(target, response, sampling_rate)

    refined_parameters = None
    refined_average_rmse = None
    if refine:
        total_steps = sum(len(rows) for rows in average_steps.values())
        if total_steps >= 3:
            print(f"Refining against {total_steps} isolated measured steps...", flush=True)
            try:
                refined_parameters, refined_average_rmse = fit_average_step_response(
                    average_time, average_steps, sampling_rate, neural_parameters
                )
            except ValueError as exc:
                print(f"Response refinement skipped: {exc}", flush=True)
        else:
            print(f"Response refinement skipped: only {total_steps} isolated steps found.", flush=True)

    if refined_parameters is not None:
        gain, natural_frequency = refined_parameters.gain, refined_parameters.natural_frequency
        damping_ratio, delay = refined_parameters.damping_ratio, refined_parameters.delay
        safe_delay = delay
        model_response = simulate_response(
            target - target[0], refined_parameters, sampling_rate=sampling_rate,
            initial_output=float(response[0]),
        )
    else:
        model_response = neural_response
    rmse = float(np.sqrt(np.mean((model_response - response) ** 2)))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    average_plot_path = output_prefix.with_name(output_prefix.name + "_average_response.png")
    step_counts = save_average_response_plot(
        average_plot_path, average_time, average_steps, neural_parameters, sampling_rate,
        refined_parameters=refined_parameters,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "gain": gain,
        "natural_frequency_rad_s": natural_frequency,
        "damping_ratio": damping_ratio,
        "delay_s": delay,
        "response_rmse_normalized": rmse,
        "neural_gain": neural_parameters.gain,
        "neural_natural_frequency_rad_s": neural_parameters.natural_frequency,
        "neural_damping_ratio": neural_parameters.damping_ratio,
        "neural_delay_s": neural_parameters.delay,
        "neural_response_rmse_normalized": neural_rmse,
        "refinement_used": refined_parameters is not None,
        "refined_average_step_rmse": refined_average_rmse,
        "lead_zero_time_s": refined_parameters.zero_time if refined_parameters is not None else 0.0,
        "duration_s": float(uniform_time[-1] - uniform_time[0]),
        "sampling_rate_hz": sampling_rate,
        "isolated_increasing_steps": step_counts["increasing"],
        "isolated_decreasing_steps": step_counts["decreasing"],
    }
    json_path = output_prefix.with_suffix(".json")
    plot_path = output_prefix.with_name(output_prefix.name + "_response.png")
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(uniform_time - uniform_time[0], target, color="0.55", linewidth=0.9)
    axes[0].set_ylabel(f"Target ({input_column}) / {scale:g}")
    axes[1].plot(uniform_time - uniform_time[0], response, label="Measured response", linewidth=0.8)
    axes[1].plot(uniform_time - uniform_time[0], neural_response, label="Neural estimate",
                 linewidth=1.0, alpha=0.8)
    if refined_parameters is not None:
        axes[1].plot(uniform_time - uniform_time[0], model_response,
                     label="Refined mean-step fit", linewidth=1.2)
    axes[1].set_ylabel(f"Response ({output_column}) / {scale:g}")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"{input_column} → {output_column}: K={gain:.3f}, wn={natural_frequency:.2f} rad/s, "
        f"zeta={damping_ratio:.3f}, delay={delay:.3f} s, "
        f"Tz={refined_parameters.zero_time if refined_parameters is not None else 0.0:.3f} s, "
        f"RMSE={rmse:.4f}"
    )
    fig.tight_layout()
    plot_path = _save_figure(fig, plot_path, dpi=160)
    plt.close(fig)

    for key, value in metrics.items():
        print(f"{key}: {value:.6g}")
    print(f"Saved metrics to {json_path}")
    print(f"Saved response plot to {plot_path}")
    print(f"Saved event-aligned average response to {average_plot_path}")
    print(f"Included steps: increasing={step_counts['increasing']}, decreasing={step_counts['decreasing']}")
    if refined_parameters is not None:
        print(
            "Refined mean-step parameters: "
            f"K={gain:.4f}, wn={natural_frequency:.4f} rad/s, "
            f"zeta={damping_ratio:.4f}, delay={safe_delay:.4f} s, "
            f"Tz={refined_parameters.zero_time:.4f} s; "
            f"mean-step RMSE={refined_average_rmse:.5f}"
        )
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
    parser.add_argument("--no-refine", action="store_true",
                        help="Use only the neural estimate; skip measured step-response refinement.")
    args = parser.parse_args()
    predict_tsv(args.model, args.tsv, args.input_column, args.output_column,
                args.output_prefix, args.scale, refine=not args.no_refine)


if __name__ == "__main__":
    main()
