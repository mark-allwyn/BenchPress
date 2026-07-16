"""Deterministic simulators for the simulation tier. Every gold state is produced
by literally running the machine here - the oracle is trivially correct (no subtle
definitions), which is the whole point after the causal-gold experience.

All grids are toroidal (edges wrap). States are small ints; rows render as digit
strings."""

from __future__ import annotations

import random

_OFFSETS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def seeded_grid(seed: int, size: int, density: float, states: int = 1):
    """Reproducible size x size grid. states=1 -> cells in {0,1}; else {0..states}."""
    rng = random.Random(seed)
    if states == 1:
        return [[1 if rng.random() < density else 0 for _ in range(size)] for _ in range(size)]
    return [[rng.randint(0, states) if rng.random() < density else 0 for _ in range(size)]
            for _ in range(size)]


def seeded_row(seed: int, width: int, density: float = 0.5):
    rng = random.Random(seed)
    return [1 if rng.random() < density else 0 for _ in range(width)]


def _live_neighbours(grid, i, j, H, W, live=1):
    return sum(1 for di, dj in _OFFSETS if grid[(i + di) % H][(j + dj) % W] == live)


def life_like(grid, gens, born, survive):
    """Life-like rule: dead cell is born iff neighbour count in `born`; live cell
    survives iff count in `survive`. born/survive are sets of ints."""
    H, W = len(grid), len(grid[0])
    for _ in range(gens):
        nxt = [[0] * W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                n = _live_neighbours(grid, i, j, H, W)
                nxt[i][j] = 1 if ((grid[i][j] and n in survive) or (not grid[i][j] and n in born)) else 0
        grid = nxt
    return grid


def brians_brain(grid, gens):
    """3-state CA. 0=off, 1=on, 2=dying. on->dying, dying->off, off->on iff exactly
    two ON (1) neighbours."""
    H, W = len(grid), len(grid[0])
    for _ in range(gens):
        nxt = [[0] * W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                c = grid[i][j]
                if c == 1:
                    nxt[i][j] = 2
                elif c == 2:
                    nxt[i][j] = 0
                else:
                    nxt[i][j] = 1 if _live_neighbours(grid, i, j, H, W, live=1) == 2 else 0
        grid = nxt
    return grid


def eca(row, rule, steps):
    """Elementary cellular automaton, cyclic boundary."""
    n = len(row)
    for _ in range(steps):
        row = [(rule >> (row[(i - 1) % n] * 4 + row[i] * 2 + row[(i + 1) % n])) & 1 for i in range(n)]
    return row


def rows_str(grid):
    return ["".join(str(c) for c in r) for r in grid]
