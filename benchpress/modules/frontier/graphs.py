"""Graph computations for the frontier tier. Every gold value here is computed
by an exact algorithm (networkx + brute force), so there is no judge and no
ambiguity - the whole benchmark rests on these being correct."""

from __future__ import annotations

import itertools
import random

import networkx as nx


def seeded_dag(seed: int, n: int, density: float):
    """A reproducible random DAG on n nodes (edges only i<j, so always acyclic)."""
    rng = random.Random(seed)
    nodes = [f"V{i + 1}" for i in range(n)]
    edges = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < density]
    G = nx.DiGraph(edges)
    G.add_nodes_from(nodes)
    return nodes, edges, G


def d_separated(G, x, y, S) -> bool:
    return nx.is_d_separator(G, {x}, {y}, set(S))


def vstructures(edges, nodes) -> set:
    pred = {n: set() for n in nodes}
    adj = {n: set() for n in nodes}
    for a, b in edges:
        pred[b].add(a)
        adj[a].add(b)
        adj[b].add(a)
    return {
        (a, c, b)
        for c in nodes
        for a, b in itertools.combinations(sorted(pred[c]), 2)
        if b not in adj[a]
    }


def vstructure_count(edges, nodes) -> int:
    return len(vstructures(edges, nodes))


def _mec_members(edges, nodes):
    skeleton = [tuple(sorted(e)) for e in edges]
    target = vstructures(edges, nodes)
    members = []
    for orient in itertools.product([0, 1], repeat=len(skeleton)):
        directed = [(u, v) if o == 0 else (v, u) for (u, v), o in zip(skeleton, orient)]
        G = nx.DiGraph(directed)
        G.add_nodes_from(nodes)
        if nx.is_directed_acyclic_graph(G) and vstructures(directed, nodes) == target:
            members.append(set(directed))
    return members, skeleton


def mec_size(edges, nodes) -> int:
    members, _ = _mec_members(edges, nodes)
    return len(members)


def compelled_count(edges, nodes) -> int:
    """Edges with the same orientation in every member of the equivalence class."""
    members, skeleton = _mec_members(edges, nodes)
    count = 0
    for u, v in skeleton:
        orientations = {((u, v) if (u, v) in m else (v, u)) for m in members}
        if len(orientations) == 1:
            count += 1
    return count


def linear_extension_count(G, cap: int | None = None) -> int:
    """Number of valid topological orderings. With cap, stop counting past it
    (sparse graphs can have astronomically many orderings)."""
    count = 0
    for _ in nx.all_topological_sorts(G):
        count += 1
        if cap is not None and count > cap:
            return count
    return count


def min_separator_size(G, x, y):
    """Smallest conditioning set that d-separates x and y (None if not separable)."""
    others = [v for v in G.nodes if v not in (x, y)]
    for size in range(len(others) + 1):
        for S in itertools.combinations(others, size):
            if d_separated(G, x, y, S):
                return size
    return None


def minimal_separators(G, x, y) -> list:
    """All minimal d-separating sets (no proper subset also separates)."""
    others = [v for v in G.nodes if v not in (x, y)]
    seps = []
    for size in range(len(others) + 1):
        for S in itertools.combinations(others, size):
            if d_separated(G, x, y, S) and not any(set(p) < set(S) for p in seps):
                seps.append(S)
    return seps


def seeded_weighted_dag(seed: int, n: int, density: float):
    """A reproducible DAG with linear structural coefficients on each edge."""
    nodes, edges, G = seeded_dag(seed, n, density)
    rng = random.Random(seed * 131 + 7)
    w = {e: rng.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) * rng.choice([-1, 1]) for e in edges}
    return nodes, edges, G, w


def total_effect(G, w, x, y) -> float:
    """Total causal effect of x on y in a linear SEM: sum over every directed path
    x->y of the product of the coefficients along it (Wright's rules)."""
    tot = 0.0
    for path in nx.all_simple_paths(G, x, y):
        prod = 1.0
        for a, b in zip(path, path[1:]):
            prod *= w[(a, b)]
        tot += prod
    return tot


def _minus_out(G, S):
    """Copy of G with every edge leaving a node in S removed."""
    Gm = G.copy()
    for s in S:
        Gm.remove_edges_from(list(G.out_edges(s)))
    return Gm


def backdoor_set_valid(G, x, y, Z) -> bool:
    """Z is a valid back-door adjustment set for X->Y iff Z has no descendant of X
    (nor Y), and Z blocks every back-door path from X to Y."""
    Z = set(Z)
    desc = nx.descendants(G, x) | {x}
    if any(z in desc or z == y for z in Z):
        return False
    return nx.is_d_separator(_minus_out(G, {x}), {x}, {y}, Z)


def frontdoor_set_valid(G, x, y, Z) -> bool:
    """Z satisfies the front-door criterion for X->Y iff (1) Z intercepts every
    directed path X->Y, (2) no unblocked back-door path X->Z, (3) every back-door
    path Z->Y is blocked by X."""
    Z = set(Z)
    if not Z or x in Z or y in Z:
        return False
    sub = G.subgraph([n for n in G.nodes if n not in Z])
    if x in sub and y in sub and nx.has_path(sub, x, y):
        return False
    if not nx.is_d_separator(_minus_out(G, {x}), {x}, Z, set()):
        return False
    if not nx.is_d_separator(_minus_out(G, Z), Z, {y}, {x}):
        return False
    return True


def instrument_valid(G, v, x, y) -> bool:
    """V is a valid instrument for X->Y iff V is associated with X (d-connected)
    and V affects Y only through X (d-separated from Y once X's out-edges are cut)."""
    if v in (x, y):
        return False
    if d_separated(G, v, x, set()):
        return False
    return nx.is_d_separator(_minus_out(G, {x}), {v}, {y}, set())


def open_path_count(G, x, y, S) -> int:
    """Number of active (d-connecting) trails between x and y given S."""
    undirected = G.to_undirected()
    anc = set(S)
    for v in S:
        anc |= nx.ancestors(G, v)
    count = 0
    for path in nx.all_simple_paths(undirected, x, y):
        active = True
        for i in range(1, len(path) - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            collider = G.has_edge(a, b) and G.has_edge(c, b)
            if (collider and b not in anc) or (not collider and b in S):
                active = False
                break
        count += active
    return count
