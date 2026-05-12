from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from tqdm import trange

from deep_ritz_lite.model import MLP
from deep_ritz_lite.plot import plot_run
from deep_ritz_lite.problem import boundary_value, exact_solution, relative_l2_error, rhs
from deep_ritz_lite.sampling import make_grid, sample_boundary, sample_interior


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_ritz_loss(
    model: torch.nn.Module,
    interior_x: torch.Tensor,
    boundary_x: torch.Tensor,
    boundary_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    interior_x = interior_x.detach().requires_grad_(True)

    u = model(interior_x)
    grad_u = torch.autograd.grad(
        u.sum(),
        interior_x,
        create_graph=True,
    )[0]

    energy = (0.5 * grad_u.pow(2).sum(dim=1, keepdim=True) - rhs(interior_x) * u).mean()

    boundary_pred = model(boundary_x)
    boundary_loss = (boundary_pred - boundary_value(boundary_x)).pow(2).mean()
    total = energy + boundary_weight * boundary_loss
    return total, energy.detach(), boundary_loss.detach()


@torch.no_grad()
def evaluate(model: torch.nn.Module, grid_n: int, device: torch.device) -> float:
    grid = make_grid(grid_n, device)
    pred = model(grid)
    truth = exact_solution(grid)
    return float(relative_l2_error(pred, truth).cpu())


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "energy", "boundary", "rel_l2"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Deep Ritz model.")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--interior", type=int, default=2048)
    parser.add_argument("--boundary", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--boundary-weight", type=float, default=100.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--grid", type=int, default=101)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["out"] = str(args.out)
    config["device"] = str(device)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = MLP(hidden=args.hidden, layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    progress = trange(1, args.epochs + 1, desc="training", dynamic_ncols=True)
    for epoch in progress:
        interior_x = sample_interior(args.interior, device)
        boundary_x = sample_boundary(args.boundary, device)

        optimizer.zero_grad(set_to_none=True)
        loss, energy, boundary = deep_ritz_loss(model, interior_x, boundary_x, args.boundary_weight)
        loss.backward()
        optimizer.step()

        rel_l2 = float("nan")
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            rel_l2 = evaluate(model, args.grid, device)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "energy": float(energy.cpu()),
                    "boundary": float(boundary.cpu()),
                    "rel_l2": rel_l2,
                }
            )
            progress.set_postfix(loss=f"{loss.item():.3e}", rel_l2=f"{rel_l2:.3e}")

    torch.save(model.state_dict(), run_dir / "model.pt")
    write_history(run_dir / "history.csv", history)

    if not args.no_plots:
        plot_run(run_dir)

    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
