"""Independent reference/prediction figures for evaluation targets."""
from pathlib import Path

import numpy as np


def save_parity_plots(pairs, metrics, directory, *, should_stop=None):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    titles = {"energy": "Energy", "forces": "Forces", "charges": "Atomic charges",
              "dipole": "Dipole", "stress": "Stress"}
    artifacts = {}
    for key, chunks in pairs.items():
        if should_stop is not None and should_stop():
            from mlpui.training import TrainingStopped
            raise TrainingStopped("Evaluation stopped by user")
        values = np.concatenate(chunks)
        reference, prediction = values.T
        low, high = float(values.min()), float(values.max())
        margin = max((high - low) * .05, max(abs(low), abs(high), 1.) * 1e-6)
        limits = (low - margin, high + margin)
        fig = Figure(figsize=(6.4, 5.4), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        density = ax.hexbin(reference, prediction, gridsize=80, extent=(*limits, *limits),
                            mincnt=1, cmap="viridis", norm=LogNorm(vmin=1))
        fig.colorbar(density, ax=ax, label="Count (log scale)")
        ax.plot(limits, limits, "--", color="crimson", linewidth=1.2)
        ax.set(xlim=limits, ylim=limits, aspect="equal", title=f"{titles.get(key, key)} prediction",
               xlabel="Reference (converted dataset units)",
               ylabel="Predicted (converted dataset units)")
        metric = metrics[key]
        ax.text(.04, .96, f"RMSE = {metric['rmse']:.5g}\nMAE = {metric['mae']:.5g}\nN = {metric['count']:,}",
                transform=ax.transAxes, va="top", bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
        artifacts[key] = {}
        for extension in ("png", "svg"):
            filename = f"{key}.{extension}"
            fig.savefig(directory / filename, dpi=180)
            artifacts[key][extension] = filename
        fig.clear()
    return artifacts
