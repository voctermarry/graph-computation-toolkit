"""Graph construction and the CSR representation.

Nodes are strings. Edges carry a numeric weight (default 1.0). Iteration order is always sorted, so
every algorithm in this package is deterministic without depending on insertion order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .errors import ValidationError


def checked_weight(weight: object) -> float:
    """The one weight rule for every construction entry point.

    Only a finite ``int`` or ``float`` is a weight: booleans, strings, ``None`` and the non-finite
    floats (NaN, Infinity, -Infinity) are rejected, because any of them would silently corrupt
    distances, rankings and the JSON output. Legal values are normalised to ``float``.
    """
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValidationError("edge weight must be a number", value=weight)
    if not math.isfinite(weight):
        raise ValidationError("edge weight must be a finite number", value=weight)
    return float(weight)


@dataclass(frozen=True, slots=True)
class Edge:
    source: str
    target: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValidationError("edge endpoints must be non-empty", value=f"{self.source!r}->{self.target!r}")
        checked_weight(self.weight)

    def to_document(self) -> dict[str, object]:
        return {"source": self.source, "target": self.target, "weight": float(self.weight)}


@dataclass(slots=True)
class Graph:
    directed: bool = False
    _adjacency: dict[str, dict[str, float]] = field(default_factory=dict, init=False)

    # -- construction ----------------------------------------------------------------------------
    def add_node(self, node: str) -> None:
        if not node:
            raise ValidationError("node name must be non-empty")
        self._adjacency.setdefault(node, {})

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        # Validate before touching any state: a rejected weight must not add endpoints, overwrite
        # an existing weight, or leave half of an undirected edge behind.
        weight = checked_weight(weight)
        self.add_node(source)
        self.add_node(target)
        if self.directed:
            self._adjacency[source][target] = weight
        else:
            self._adjacency[source][target] = weight
            self._adjacency[target][source] = weight

    def remove_edge(self, source: str, target: str) -> bool:
        removed = False
        if source in self._adjacency and target in self._adjacency[source]:
            del self._adjacency[source][target]
            removed = True
        if not self.directed and target in self._adjacency and source in self._adjacency[target]:
            del self._adjacency[target][source]
            removed = True
        return removed

    @classmethod
    def from_edges(cls, edges: list[Edge], *, directed: bool = False) -> "Graph":
        graph = cls(directed=directed)
        for edge in edges:
            graph.add_edge(edge.source, edge.target, edge.weight)
        return graph

    # -- queries ---------------------------------------------------------------------------------
    @property
    def node_count(self) -> int:
        return len(self._adjacency)

    def nodes(self) -> list[str]:
        return sorted(self._adjacency)

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        if node not in self._adjacency:
            raise ValidationError(f"unknown node: {node}", known=self.nodes()[:10])
        return [(target, self._adjacency[node][target]) for target in sorted(self._adjacency[node])]

    def edges(self) -> list[Edge]:
        seen: set[tuple[str, str]] = set()
        result: list[Edge] = []
        for source in self.nodes():
            for target in sorted(self._adjacency[source]):
                if not self.directed and (target, source) in seen:
                    continue
                seen.add((source, target))
                result.append(Edge(source, target, self._adjacency[source][target]))
        return result

    def edge_count(self) -> int:
        return len(self.edges())

    def degree(self, node: str) -> int:
        if node not in self._adjacency:
            raise ValidationError(f"unknown node: {node}")
        return len(self._adjacency[node])

    def weighted_degree(self, node: str) -> float:
        self.neighbors(node)
        return sum(self._adjacency[node].values())

    def self_loops(self) -> list[str]:
        return [node for node in self.nodes() if node in self._adjacency[node]]

    def has_negative_weights(self) -> bool:
        return any(edge.weight < 0 for edge in self.edges())

    def csr(self) -> tuple[list[str], list[int], list[tuple[str, float]]]:
        """Compressed sparse row: node order, offsets, and the flattened neighbour list."""
        nodes = self.nodes()
        offsets = [0]
        flat: list[tuple[str, float]] = []
        for node in nodes:
            flat.extend(self.neighbors(node))
            offsets.append(len(flat))
        return nodes, offsets, flat

    def to_document(self) -> dict[str, object]:
        return {
            "directed": self.directed,
            "nodes": self.node_count,
            "edges": self.edge_count(),
            "selfLoops": self.self_loops(),
            "negativeWeights": self.has_negative_weights(),
        }
