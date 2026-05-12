# deep-ritz-lite

A lightweight PyTorch implementation of the Deep Ritz method for a 2D Poisson
equation. The repository is intentionally small so it is easy to read, modify,
and use as a starting point for experiments.

Problem:

```text
-Delta u = f,  x in (0, 1)^2
u = 0,         x on boundary
```

with exact solution:

```text
u(x, y) = sin(pi x) sin(pi y)
f(x, y) = 2 pi^2 sin(pi x) sin(pi y)
```

The Deep Ritz loss minimizes the energy

```text
E(u) = integral(0.5 |grad u|^2 - f u) dx
```

plus a boundary penalty.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python -m deep_ritz_lite.train --epochs 3000 --hidden 64 --layers 4
```

Outputs are written to `runs/<timestamp>/`:

- `config.json`
- `history.csv`
- `model.pt`
- `loss.png`
- `solution.png`
- `error.png`

## Plot An Existing Run

```bash
python -m deep_ritz_lite.plot --run runs/<timestamp>
```

## Repository Layout

```text
deep-ritz-lite/
├─ deep_ritz_lite/
│  ├─ __init__.py
│  ├─ model.py
│  ├─ problem.py
│  ├─ sampling.py
│  ├─ train.py
│  └─ plot.py
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## Next Experiments

- Change the exact solution in `problem.py`.
- Try different boundary penalties with `--boundary-weight`.
- Increase dimensions by changing the sampler and problem definition.
- Compare Deep Ritz with a residual loss `||-Delta u - f||^2`.
