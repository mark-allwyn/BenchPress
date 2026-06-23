"""B16 proxy confounder. A latent confounder has an observed proxy (a noisy
child). Adjusting for the proxy does not block the backdoor path, so the effect
stays unidentifiable - a common 'we controlled for a related variable' error.
Verified with networkx."""

from __future__ import annotations

import random

import networkx as nx

from benchpress.core.types import Item, Part
from benchpress.modules.causal import dag


def _build() -> dict:
    # U latent confounder of X,Y; P observed proxy caused by U; X->Y.
    edges = [("_u", "_x"), ("_u", "_y"), ("_u", "_p"), ("_x", "_y")]
    return {"edges": edges, "x": "_x", "y": "_y", "latent": "_u", "proxy": "_p"}


def _relabel(rng: random.Random, raw: dict) -> dict:
    nodes = sorted({n for e in raw["edges"] for n in e})
    labels = [f"V{i + 1}" for i in range(len(nodes))]
    rng.shuffle(labels)
    m = dict(zip(nodes, labels))
    return {
        "edges": sorted([m[a], m[b]] for a, b in raw["edges"]),
        "x": m[raw["x"]], "y": m[raw["y"]],
        "latent": m[raw["latent"]], "proxy": m[raw["proxy"]],
        "observed": sorted(m[n] for n in nodes if n != raw["latent"]),
    }


def _render(p: dict) -> str:
    edge_lines = ", ".join(f"{a}->{b}" for a, b in p["edges"])
    return f"""Consider this causal diagram (directed edges):
{edge_lines}

Variable {p['latent']} is unobserved (latent). {p['proxy']} is an observed proxy caused by {p['latent']}. All variables except {p['latent']} are observed. We want the causal effect of {p['x']} on {p['y']}.

Determine:
1. whether adjusting for the proxy {p['proxy']} alone is sufficient to identify the effect (yes or no);
2. whether the effect is identifiable by adjusting for any set of observed variables (yes or no).

ANSWER FORMAT - reply with exactly these labelled lines and nothing else:
PROXY_SUFFICIENT: <yes or no>
IDENTIFIABLE_BY_ADJUSTMENT: <yes or no>

WORKED EXAMPLE (unrelated graph, format only):
PROXY_SUFFICIENT: no
IDENTIFIABLE_BY_ADJUSTMENT: no"""


def make_item(rng: random.Random, draw: int, version: str) -> Item:
    p = _relabel(rng, _build())
    G = nx.DiGraph([tuple(e) for e in p["edges"]])
    proxy_ok = "yes" if dag.satisfies_backdoor(G, p["x"], p["y"], {p["proxy"]}) else "no"
    by_adj = "yes" if dag.identifiable_by_adjustment(G, p["x"], p["y"], p["observed"]) else "no"
    parts = [
        Part("PROXY_SUFFICIENT", "categorical", proxy_ok, {"vocab": ["yes", "no"]}, ["proxy"]),
        Part("IDENTIFIABLE_BY_ADJUSTMENT", "categorical", by_adj, {"vocab": ["yes", "no"]}, ["proxy", "identification"]),
    ]
    return Item(
        item_id=f"causal-v{version}-B16-{draw:04d}",
        module="causal", bundle_id="B16", variant="transfer", difficulty="hard",
        gen_params=p, prompt=_render(p), parts=parts,
        skill_tags=["proxy", "identification"],
    )


def verify_proxy(item: Item) -> bool:
    p = item.gen_params
    G = nx.DiGraph([tuple(e) for e in p["edges"]])
    parts = {pt.part_id: pt for pt in item.parts}
    proxy_ok = "yes" if dag.satisfies_backdoor(G, p["x"], p["y"], {p["proxy"]}) else "no"
    by_adj = "yes" if dag.identifiable_by_adjustment(G, p["x"], p["y"], p["observed"]) else "no"
    if str(parts["PROXY_SUFFICIENT"].expected).lower() != proxy_ok:
        return False
    if str(parts["IDENTIFIABLE_BY_ADJUSTMENT"].expected).lower() != by_adj:
        return False
    # the teaching case: proxy is not sufficient and adjustment cannot identify
    return proxy_ok == "no" and by_adj == "no"
