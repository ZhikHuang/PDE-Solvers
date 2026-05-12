from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from deep_ritz_lite.model import MLP
from deep_ritz_lite.problem import exact_solution
from deep_ritz_lite.sampling import make_grid


def read_history(path: Path) -> dict[str, list[float]]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    return {
        key: [float(row[key]) for row in rows]
        for key in ["epoch", "loss", "energy", "boundary", "rel_l2"]
    }


def plot_history(run_dir: Path) -> None:
    history = read_history(run_dir / "history.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["loss"], label="total")
    axes[0].plot(history["epoch"], history["energy"], label="energy")
    axes[0].plot(history["epoch"], history["boundary"], label="boundary")
    axes[0].set_yscale("symlog")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["rel_l2"])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("relative L2 error")

    fig.tight_layout()
    fig.savefig(run_dir / "loss.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def plot_solution(run_dir: Path) -> None:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    grid_n = int(config.get("grid", 101))

    model = MLP(hidden=int(config["hidden"]), layers=int(config["layers"])).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()

    grid = make_grid(grid_n, device)
    pred = model(grid).reshape(grid_n, grid_n).cpu()
    truth = exact_solution(grid).reshape(grid_n, grid_n).cpu()
    error = torch.abs(pred - truth)

    extent = [0.0, 1.0, 0.0, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(pred, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title("Deep Ritz prediction")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(truth, origin="lower", extent=extent, cmap="viridis")
    axes[1].set_title("exact solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    fig.tight_layout()
    fig.savefig(run_dir / "solution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(error, origin="lower", extent=extent, cmap="magma")
    ax.set_title("absolute error")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(run_dir / "error.png", dpi=160)
    plt.close(fig)


def plot_run(run_dir: str | Path) -> None:
    run_path = Path(run_dir)
    plot_history(run_path)
    plot_solution(run_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a deep-ritz-lite run.")
    parser.add_argument("--run", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_run(args.run)
    print(f"Plots written to: {args.run}")


if __name__ == "__main__":
    main()
