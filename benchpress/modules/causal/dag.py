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


def is_instrument(G: nx.DiGraph, z, x, y) -> bool:
    """Graphical instrument test: Z is relevant to X (d-connected) and affects Y
    only through X (d-separated from Y once X's outgoing edges are removed)."""
    if z in (x, y):
        return False
    if nx.is_d_separator(G, {z}, {x}, set()):  # not relevant
        return False
    H = G.copy()
    H.remove_edges_from(list(G.out_edges(x)))
    return nx.is_d_separator(H, {z}, {y}, set())


def is_front_door(G: nx.DiGraph, M, x, y) -> bool:
    """Front-door criterion for set M relative to (X, Y):
    (i) M intercepts every directed path X->Y;
    (ii) no unblocked backdoor path X->M;
    (iii) every backdoor path M->Y is blocked by X."""
    M = set(M)
    if x in M or y in M:
        return False
    without_m = G.copy()
    without_m.remove_nodes_from(M)
    if nx.has_path(without_m, x, y):
        return False
    gx = G.copy()
    gx.remove_edges_from(list(G.out_edges(x)))
    if not nx.is_d_separator(gx, {x}, M, set()):
        return False
    gm = G.copy()
    for m in M:
        gm.remove_edges_from(list(G.out_edges(m)))
    return nx.is_d_separator(gm, M, {y}, {x})


def identifiable_by_adjustment(G: nx.DiGraph, x, y, observed) -> bool:
    """Whether some subset of OBSERVED variables is a valid backdoor set."""
    excluded = nx.descendants(G, x) | {x, y}
    candidates = [n for n in observed if n not in excluded]
    for r in range(len(candidates) + 1):
        for combo in combinations(candidates, r):
            if satisfies_backdoor(G, x, y, set(combo)):
                return True
    return False
