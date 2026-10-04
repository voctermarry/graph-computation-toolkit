"""Graph algorithms: traversal, shortest paths, components, ordering, PageRank, centrality, clustering.

Every function is deterministic (sorted iteration), reports unreachable nodes explicitly rather than
inventing a distance, and raises a typed error with evidence when the input makes the answer
undefined -- a negative edge for Dijkstra, a negative cycle for Bellman-Ford, a cycle for a
topological sort.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    # Nodes first: a node with no incident edge (added bare, or left behind by remove_edge) must
    # still appear as its own component -- rebuilding from edges alone would silently drop it.
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
    """Power iteration with explicit convergence reporting (never a silent iteration cap)."""
    if not 0.0 < damping < 1.0:
        raise ValidationError("damping must be in (0, 1)", value=damping)
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
