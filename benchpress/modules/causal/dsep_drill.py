"""B12 conditional-independence drill (multi-query d-separation). A random DAG
with three d-separation queries; the item is correct only if all three verdicts
are right. Every answer is computed (and re-checked in the gate) by networkx."""

from __future__ import annotations

import random

import networkx as nx

from benchpress.core.types import Item, Part


def _random_dag(rng: random.Random, n: int = 6, p: float = 0.45):
    nodes = [f"V{i + 1}" for i in range(n)]
    for _ in range(200):
        edges = [
            (nodes[i], nodes[j])
            for i in range(n) for j in range(i + 1, n)
            if rng.random() < p
        ]
        if 3 <= len(edges) <= n + 3:
            return nodes, edges
    return nodes, edges  # fall back to whatever the last draw produced


def _queries(rng: random.Random, nodes: list[str], G: nx.DiGraph) -> list[dict]:
    qs = []
    seen = set()
    for _ in range(400):
        if len(qs) == 3:
            break
        a, b = rng.sample(nodes, 2)
        rest = [v for v in nodes if v not in (a, b)]
        k = rng.randint(0, min(2, len(rest)))
        cond = sorted(rng.sample(rest, k)) if k else []
        key = (a, b, tuple(cond))
        if key in seen:
            continue
        seen.add(key)
        ans = "yes" if nx.is_d_separator(G, {a}, {b}, set(cond)) else "no"
        qs.append({"a": a, "b": b, "cond": cond, "answer": ans})
    return qs


def _render(p: dict) -> str:
    edge_lines = ", ".join(f"{a}->{b}" for a, b in p["edges"])
    lines = []
    for i, q in enumerate(p["queries"], 1):
        cond = "{" + ", ".join(q["cond"]) + "}"
        lines.append(f"Q{i}: are {q['a']} and {q['b']} d-separated given {cond}?")
    queries = "\n".join(lines)
    return f"""Consider this causal diagram (directed edges):
{edge_lines}

Answer each d-separation query with yes or no:
{queries}

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
Q1: <yes or no>
Q2: <yes or no>
Q3: <yes or no>

WORKED EXAMPLE (unrelated graph, format only):
Q1: yes
Q2: no
Q3: yes"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    nodes, edges = _random_dag(rng)
    G = nx.DiGraph(edges)
    G.add_nodes_from(nodes)
    queries = _queries(rng, nodes, G)
    p = {"edges": sorted([list(e) for e in edges]), "nodes": nodes, "queries": queries}
    parts = [
        Part(f"Q{i}", "categorical", q["answer"], {"vocab": ["yes", "no"]}, ["d_separation"])
        for i, q in enumerate(queries, 1)
    ]
    return Item(
        item_id=f"causal-v{version}-B12-{draw:04d}",
        module="causal", bundle_id="B12", variant="transfer", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["d_separation"],
    )


def verify_dsep_drill(item: Item) -> bool:
    p = item.gen_params
    G = nx.DiGraph([tuple(e) for e in p["edges"]])
    G.add_nodes_from(p["nodes"])
    parts = {pt.part_id: pt for pt in item.parts}
    for i, q in enumerate(p["queries"], 1):
        ans = "yes" if nx.is_d_separator(G, {q["a"]}, {q["b"]}, set(q["cond"])) else "no"
        if str(parts[f"Q{i}"].expected).lower() != ans:
            return False
    return True
