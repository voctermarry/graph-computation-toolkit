"""Graph algorithms: traversal, shortest paths, components, ordering, PageRank, centrality, clustering, communities.

Every function is deterministic (sorted iteration), reports unreachable nodes explicitly rather than
inventing a distance, and raises a typed error with evidence when the input makes the answer
undefined -- a negative edge for Dijkstra, a negative cycle for Bellman-Ford, a cycle for a
topological sort.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping

from .errors import CycleError, NegativeCycleError, NegativeWeightError, ValidationError
from .graph import Graph

INFINITY = math.inf


def _check_source(graph: Graph, source: str) -> None:
    if source not in graph.nodes():
        raise ValidationError(f"unknown source: {source}", known=graph.nodes()[:10])


def bfs(graph: Graph, source: str) -> dict[str, int]:
    """Hop distances from `source`; unreachable nodes are absent (never faked as -1)."""
    _check_source(graph, source)
    seen = {source: 0}
    queue = [source]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for neighbour, _ in graph.neighbors(node):
            if neighbour not in seen:
                seen[neighbour] = seen[node] + 1
                queue.append(neighbour)
    return seen


def dfs(graph: Graph, source: str) -> list[str]:
    """Pre-order depth-first walk, neighbours visited in sorted order."""
    _check_source(graph, source)
    visited: list[str] = []
    seen: set[str] = set()
    stack = [source]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        visited.append(node)
        stack.extend(neighbour for neighbour, _ in reversed(graph.neighbors(node)) if neighbour not in seen)
    return visited


def dijkstra(graph: Graph, source: str) -> tuple[dict[str, float], dict[str, str | None]]:
    """Non-negative shortest paths; a negative edge is refused, not silently mis-answered.

    The check walks every edge once, up front: discovering a negative weight halfway through the
    relaxation loop would mean returning a distance that is simply wrong.
    """
    _check_source(graph, source)
    for edge in graph.edges():
        if edge.weight < 0:
            raise NegativeWeightError(
                "dijkstra requires non-negative weights; use bellman_ford",
                edge=f"{edge.source}->{edge.target}",
                weight=edge.weight,
            )
    distance: dict[str, float] = {source: 0.0}
    previous: dict[str, str | None] = {source: None}
    remaining = set(graph.nodes())
    while remaining:
        current = None
        best = INFINITY
        for node in sorted(remaining):
            if node in distance and distance[node] < best:
                current, best = node, distance[node]
        if current is None:
            break
        remaining.discard(current)
        for neighbour, weight in graph.neighbors(current):
            candidate = distance[current] + weight
            if neighbour not in distance or candidate < distance[neighbour]:
                distance[neighbour] = candidate
                previous[neighbour] = current
    return distance, previous


def bellman_ford(graph: Graph, source: str) -> tuple[dict[str, float], dict[str, str | None]]:
    """Shortest paths with negative weights; reports the offending cycle when one exists."""
    _check_source(graph, source)
    distance: dict[str, float] = {source: 0.0}
    previous: dict[str, str | None] = {source: None}
    nodes = graph.nodes()
    for _ in range(len(nodes) - 1):
        changed = False
        for node in nodes:
            if node not in distance:
                continue
            for neighbour, weight in graph.neighbors(node):
                candidate = distance[node] + weight
                if neighbour not in distance or candidate < distance[neighbour]:
                    distance[neighbour] = candidate
                    previous[neighbour] = node
                    changed = True
        if not changed:
            break
    for node in nodes:
        if node not in distance:
            continue
        for neighbour, weight in graph.neighbors(node):
            if neighbour not in distance or distance[node] + weight < distance[neighbour]:
                raise NegativeCycleError("negative cycle reachable from the source", at=f"{node}->{neighbour}")
    return distance, previous


def path_from(previous: dict[str, str | None], target: str) -> list[str]:
    """Reconstruct a path from a predecessor map; raises when the target is unreachable."""
    if target not in previous:
        raise ValidationError(f"unreachable target: {target}")
    path = [target]
    while previous[path[-1]] is not None:
        path.append(str(previous[path[-1]]))
    return list(reversed(path))


def _validated_heuristic(graph: Graph, target: str, heuristic: Mapping[str, float] | None) -> dict[str, float]:
    """Normalise the heuristic map, or raise ValidationError naming the first offending entry.

    Nodes missing from the map estimate to 0. Every supplied value must be a finite, non-negative
    number naming a node of the graph; the target must estimate to exactly 0; and along every
    traversable direction the estimate must be consistent -- h(u) <= weight(u, v) + h(v) -- which is
    what guarantees the first time the target leaves the frontier its distance is final. All checks
    walk nodes and edges in sorted order, so the reported violation never depends on the mapping's
    key order.
    """
    estimates = {node: 0.0 for node in graph.nodes()}
    if not heuristic:
        return estimates
    known = set(graph.nodes())
    for node in sorted(heuristic):
        value = heuristic[node]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValidationError("heuristic values must be finite numbers", node=node)
        estimate = float(value)
        if estimate < 0:
            raise ValidationError("heuristic values must be non-negative", node=node, value=estimate)
        if node not in known:
            raise ValidationError(f"heuristic names an unknown node: {node}", known=graph.nodes()[:10])
        estimates[node] = estimate
    if estimates[target] != 0.0:
        raise ValidationError("heuristic for the target must be 0", target=target, value=estimates[target])
    # A small tolerance keeps float summation noise (0.1 + 0.2) from reading as a violation; a real
    # inconsistency exceeds it by construction.
    for node in graph.nodes():
        for neighbour, weight in graph.neighbors(node):
            if estimates[node] > weight + estimates[neighbour] + 1e-9:
                raise ValidationError(
                    "heuristic is not consistent: h(u) must not exceed weight(u, v) + h(v)",
                    edge=f"{node}->{neighbour}",
                    estimate=estimates[node],
                    bound=round(weight + estimates[neighbour], 10),
                )
    return estimates


def astar(
    graph: Graph,
    source: str,
    target: str,
    heuristic: Mapping[str, float] | None = None,
) -> tuple[float, dict[str, str | None], int]:
    """Goal-directed shortest path from `source` to `target` guided by a consistent heuristic.

    Returns the distance to `target`, a predecessor map `path_from` can rebuild the route from, and
    the number of distinct nodes expanded. The frontier is a heap keyed by (estimated total cost,
    node name), so equal-cost candidates and equal-cost paths are always settled by node name; with
    an empty heuristic the search degenerates to Dijkstra and agrees with it exactly. A negative
    edge is refused up front, an unreachable target raises rather than inventing a distance.
    """
    _check_source(graph, source)
    if target not in graph.nodes():
        raise ValidationError(f"unknown target: {target}", known=graph.nodes()[:10])
    for edge in graph.edges():
        if edge.weight < 0:
            raise NegativeWeightError(
                "astar requires non-negative weights; use bellman_ford",
                edge=f"{edge.source}->{edge.target}",
                weight=edge.weight,
            )
    estimates = _validated_heuristic(graph, target, heuristic)
    distance: dict[str, float] = {source: 0.0}
    previous: dict[str, str | None] = {source: None}
    frontier: list[tuple[float, str]] = [(estimates[source], source)]
    expanded: set[str] = set()
    while frontier:
        _, current = heapq.heappop(frontier)
        if current in expanded:
            continue
        expanded.add(current)
        if current == target:
            return distance[target], previous, len(expanded)
        for neighbour, weight in graph.neighbors(current):
            if neighbour in expanded:
                continue
            candidate = distance[current] + weight
            if neighbour not in distance or candidate < distance[neighbour]:
                distance[neighbour] = candidate
                previous[neighbour] = current
                heapq.heappush(frontier, (candidate + estimates[neighbour], neighbour))
    raise ValidationError(f"unreachable target: {target}")


def components(graph: Graph) -> list[list[str]]:
    """Weakly connected components, each sorted, the list ordered by its smallest node."""
    undirected = graph if not graph.directed else _as_undirected(graph)
    seen: set[str] = set()
    groups: list[list[str]] = []
    for node in undirected.nodes():
        if node in seen:
            continue
        group = sorted(bfs(undirected, node))
        seen.update(group)
        groups.append(group)
    return groups


def _as_undirected(graph: Graph) -> Graph:
    mirror = Graph(directed=False)
    # Seed every node before mirroring the edges: a node with no incident arc -- added on its own via
    # add_node or orphaned by remove_edge -- cannot be recovered from the edge list, yet it must
    # survive the undirected reading as a singleton component.
    for node in graph.nodes():
        mirror.add_node(node)
    for edge in graph.edges():
        mirror.add_edge(edge.source, edge.target, edge.weight)
    return mirror


def topological_sort(graph: Graph) -> list[str]:
    """Kahn's algorithm; a cycle is reported with the nodes that could never be released."""
    if not graph.directed:
        raise ValidationError("topological sort requires a directed graph")
    in_degree = {node: 0 for node in graph.nodes()}
    for node in graph.nodes():
        for neighbour, _ in graph.neighbors(node):
            in_degree[neighbour] += 1
    ready = sorted(node for node, degree in in_degree.items() if degree == 0)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for neighbour, _ in graph.neighbors(node):
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                ready.append(neighbour)
                ready.sort()
    if len(order) != graph.node_count:
        stuck = sorted(node for node, degree in in_degree.items() if degree > 0)
        raise CycleError("directed cycle prevents a topological order", cycle=stuck)
    return order


@dataclass(frozen=True, slots=True)
class PageRankResult:
    scores: dict[str, float]
    iterations: int
    converged: bool

    def to_document(self) -> dict[str, object]:
        return {"iterations": self.iterations, "converged": self.converged, "scores": {node: round(value, 10) for node, value in sorted(self.scores.items())}}


def pagerank(graph: Graph, *, damping: float = 0.85, tolerance: float = 1e-9, max_iterations: int = 200) -> PageRankResult:
    """Power iteration with explicit convergence reporting (never a silent iteration cap).

    Edge weights are transition shares, so a negative one would yield negative scores, a zero
    outgoing-weight sum that divides through to nonsense, or a total that is not one. The whole
    graph participates in the redistribution -- an edge in an isolated component is not exempt -- so
    every edge is checked up front in the graph's stable edge order and a negative weight is refused
    with that edge as evidence; a zero total is fine, it simply marks a dangling node.
    """
    if not 0.0 < damping < 1.0:
        raise ValidationError("damping must be in (0, 1)", value=damping)
    for edge in graph.edges():
        if edge.weight < 0:
            raise NegativeWeightError(
                "pagerank requires non-negative edge weights",
                edge=f"{edge.source}->{edge.target}",
                weight=edge.weight,
            )
    nodes = graph.nodes()
    if not nodes:
        return PageRankResult({}, 0, True)
    total = len(nodes)
    scores = {node: 1.0 / total for node in nodes}
    outgoing = {node: sum(weight for _, weight in graph.neighbors(node)) for node in nodes}
    # Incoming edges are collected once, outside the iteration loop: rebuilding the neighbour mapping
    # inside it was both quadratic and wasteful (the first version of this function did exactly that).
    incoming_edges: dict[str, list[tuple[str, float]]] = {node: [] for node in nodes}
    for node in nodes:
        for neighbour, weight in graph.neighbors(node):
            incoming_edges[neighbour].append((node, weight))
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        dangling = sum(scores[node] for node in nodes if outgoing[node] == 0)
        updated: dict[str, float] = {}
        for node in nodes:
            incoming = 0.0
            for other, weight in incoming_edges[node]:
                if outgoing[other] > 0:
                    incoming += scores[other] * weight / outgoing[other]
            updated[node] = (1.0 - damping) / total + damping * (incoming + dangling / total)
        delta = sum(abs(updated[node] - scores[node]) for node in nodes)
        scores = updated
        if delta < tolerance:
            converged = True
            break
    return PageRankResult(scores, iterations, converged)


def degree_centrality(graph: Graph) -> dict[str, float]:
    """Degree normality: degree divided by the largest possible degree in an undirected reading."""
    total = graph.node_count
    if total <= 1:
        return {node: 0.0 for node in graph.nodes()}
    denominator = float(total - 1)
    return {node: round(graph.degree(node) / denominator, 10) for node in graph.nodes()}


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    coefficients: dict[str, float]
    average: float
    transitivity: float
    triangles: int
    connected_triples: int

    def to_document(self) -> dict[str, object]:
        return {
            "coefficients": {node: self.coefficients[node] for node in sorted(self.coefficients)},
            "average": self.average,
            "transitivity": self.transitivity,
            "triangles": self.triangles,
            "connectedTriples": self.connected_triples,
            "nodes": len(self.coefficients),
        }


def clustering(graph: Graph) -> ClusteringResult:
    """Local clustering coefficients, their average, and the global transitivity.

    Everything is measured on a simple undirected reading of the graph: an edge in either direction
    counts once, weights and duplicate adjacencies are ignored, and self-loops are neighbours of
    nobody. Every node -- including isolated ones -- is present in the result and counts towards the
    average. Each unordered triangle is counted once; a connected triple (wedge) is counted at its
    centre as a pair of distinct neighbours. All iteration is sorted, so the answer never depends on
    insertion order or set traversal.
    """
    # Built directly instead of via _as_undirected: that helper mirrors edges only, so an isolated
    # node in a directed graph would silently drop out of the average.
    adjacency: dict[str, set[str]] = {node: set() for node in graph.nodes()}
    for edge in graph.edges():
        if edge.source == edge.target:
            continue
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    nodes = sorted(adjacency)
    coefficients: dict[str, float] = {}
    coefficient_sum = 0.0
    closed_pairs = 0
    wedges = 0
    for node in nodes:
        neighbours = sorted(adjacency[node])
        degree = len(neighbours)
        pairs = degree * (degree - 1) // 2
        if degree < 2:
            coefficients[node] = 0.0
            continue
        # A linked pair of neighbours closes one wedge at this node; each triangle contributes at all
        # three of its vertices, so the total is three times the triangle count.
        between = sum(1 for index, left in enumerate(neighbours) for right in neighbours[index + 1 :] if right in adjacency[left])
        value = between / pairs
        coefficients[node] = round(value, 10)
        coefficient_sum += value
        closed_pairs += between
        wedges += pairs
    triangles = closed_pairs // 3
    average = round(coefficient_sum / len(nodes), 10) if nodes else 0.0
    transitivity = round(3 * triangles / wedges, 10) if wedges else 0.0
    return ClusteringResult(coefficients, average, transitivity, triangles, wedges)


@dataclass(frozen=True, slots=True)
class CommunitiesResult:
    communities: list[list[str]]
    labels: dict[str, str]
    iterations: int
    converged: bool

    @property
    def count(self) -> int:
        return len(self.communities)

    def to_document(self) -> dict[str, object]:
        return {
            "communities": self.communities,
            "labels": {node: self.labels[node] for node in sorted(self.labels)},
            "count": self.count,
            "iterations": self.iterations,
            "converged": self.converged,
        }


def label_propagation(graph: Graph, *, max_iterations: int = 100) -> CommunitiesResult:
    """Deterministic weighted label propagation over a simple undirected reading of the graph.

    The reading ignores self-loops and sums arcs between the same pair: an undirected edge
    contributes its weight once in each direction, and mutual arcs of a directed graph add up. Every
    node starts labelled with its own name; each round walks nodes in name order and moves a node to
    the neighbouring label with the highest summed edge weight, keeping the current label when it
    ties for the best score and otherwise preferring the lexicographically smallest tied label. A
    round with no change is convergence; hitting `max_iterations` first is reported honestly as
    `converged=False` with the deterministic labelling reached so far. Communities are the groups
    sharing a final label, each sorted by node name, the list ordered by each group's smallest node,
    and every output label is normalised to that smallest member -- so isolated nodes stay singleton
    communities and the document is byte-identical however the edges were ordered. Weights must be
    non-negative: the first negative edge in the graph's stable edge order is refused with that edge
    as evidence.
    """
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValidationError("max_iterations must be an integer >= 1", value=max_iterations)
    for edge in graph.edges():
        if edge.weight < 0:
            raise NegativeWeightError(
                "label propagation requires non-negative edge weights",
                edge=f"{edge.source}->{edge.target}",
                weight=edge.weight,
            )
    nodes = graph.nodes()
    if not nodes:
        return CommunitiesResult([], {}, 0, True)
    adjacency: dict[str, dict[str, float]] = {node: {} for node in nodes}
    for edge in graph.edges():
        if edge.source == edge.target:
            continue
        adjacency[edge.source][edge.target] = adjacency[edge.source].get(edge.target, 0.0) + edge.weight
        adjacency[edge.target][edge.source] = adjacency[edge.target].get(edge.source, 0.0) + edge.weight
    labels = {node: node for node in nodes}
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        changed = False
        for node in nodes:
            scores: dict[str, float] = {}
            for neighbour in sorted(adjacency[node]):
                label = labels[neighbour]
                scores[label] = scores.get(label, 0.0) + adjacency[node][neighbour]
            if not scores:
                continue
            best = max(scores.values())
            tied = {label for label, score in scores.items() if score == best}
            if labels[node] in tied:
                continue
            labels[node] = min(tied)
            changed = True
        if not changed:
            converged = True
            break
    groups: dict[str, list[str]] = {}
    for node in nodes:
        groups.setdefault(labels[node], []).append(node)
    # Members were appended in sorted node order, and lexicographic list ordering is then exactly
    # "ordered by the smallest member", which is the required community order.
    communities = sorted(groups.values())
    normalised = {node: members[0] for members in communities for node in members}
    return CommunitiesResult(communities, normalised, iterations, converged)
