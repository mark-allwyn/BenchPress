"""DAG reasoning for the transfer bundle: backdoor criterion + minimal sets.

Uses networkx d-separation to verify adjustment sets independently of how an
item was constructed (the two-method check behind the dual-verification gate).
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx


def _candidates(G: nx.DiGraph, x, y) -> list:
    excluded = nx.descendants(G, x) | {x, y}
    return [n for n in G.nodes if n not in excluded]


def satisfies_backdoor(G: nx.DiGraph, x, y, z) -> bool:
    """True if Z is a valid backdoor adjustment set for the effect of X on Y."""
    z = set(z)
    if z & (nx.descendants(G, x) | {x, y}):
        return False
    H = G.copy()
    H.remove_edges_from(list(G.out_edges(x)))  # the backdoor graph
    return nx.is_d_separator(H, {x}, {y}, z)


def minimal_backdoor_set(G: nx.DiGraph, x, y) -> frozenset | None:
    """Smallest valid backdoor adjustment set (searched by ascending size)."""
    candidates = _candidates(G, x, y)
    for r in range(len(candidates) + 1):
        for combo in combinations(candidates, r):
            if satisfies_backdoor(G, x, y, set(combo)):
                return frozenset(combo)
    return None


def is_minimal_backdoor(G: nx.DiGraph, x, y, z) -> bool:
    z = set(z)
    if not satisfies_backdoor(G, x, y, z):
        return False
    return all(not satisfies_backdoor(G, x, y, z - {n}) for n in z)
