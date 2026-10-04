"""Incremental maintenance of weakly connected components.

Edge insertions update the labelling in near-constant time with a union-find structure. Edge
**removals** cannot be undone that way, so instead of pretending otherwise the structure marks itself
stale and says which algorithms must be recomputed -- an honest incremental API reports what it no
longer knows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ValidationError
from .graph import Graph


@dataclass(slots=True)
class IncrementalComponents:
    """Union-find over an undirected reading of a graph, with staleness tracking."""

    graph: Graph
    _parent: dict[str, str] = field(default_factory=dict, init=False)
    _rank: dict[str, int] = field(default_factory=dict, init=False)
    _stale: set[str] = field(default_factory=set, init=False)
    revision: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.graph.directed:
            raise ValidationError("incremental components follow the undirected reading; pass a directed graph through as_undirected")
        for node in self.graph.nodes():
            self._parent[node] = node
            self._rank[node] = 0
        for edge in self.graph.edges():
            self._union(edge.source, edge.target)
        self.revision = 1

    # -- union-find ------------------------------------------------------------------------------
    def _find(self, node: str) -> str:
        self._parent.setdefault(node, node)
        while self._parent[node] != node:
            self._parent[node] = self._parent[self._parent[node]]
            node = self._parent[node]
        return node

    def _union(self, left: str, right: str) -> bool:
        root_left, root_right = self._find(left), self._find(right)
        if root_left == root_right:
            return False
        if self._rank[root_left] < self._rank[root_right]:
            root_left, root_right = root_right, root_left
        self._parent[root_right] = root_left
        if self._rank[root_left] == self._rank[root_right]:
            self._rank[root_left] += 1
        return True

    # -- maintenance -----------------------------------------------------------------------------
    def add_edge(self, source: str, target: str, weight: float = 1.0) -> bool:
        """Insert an edge; returns True when two components merged."""
        self.graph.add_edge(source, target, weight)
        merged = self._union(source, target)
        self.revision += 1
        if merged:
            self._stale.clear()
        return merged

    def remove_edge(self, source: str, target: str) -> bool:
        """Remove an edge. Union-find cannot split, so the labelling becomes stale for those nodes."""
        removed = self.graph.remove_edge(source, target)
        if removed:
            self.revision += 1
            self._stale.update((source, target))
        return removed

    @property
    def is_stale(self) -> bool:
        return bool(self._stale)

    def stale_nodes(self) -> list[str]:
        return sorted(self._stale)

    # -- answers ---------------------------------------------------------------------------------
    def labels(self) -> dict[str, str]:
        """Component label per node (the smallest member's name), or an error if a removal happened."""
        if self._stale:
            raise ValidationError(
                "components are stale after an edge removal; rebuild or recompute",
                stale=self.stale_nodes()[:10],
            )
        return {node: self._canonical(self._find(node)) for node in self.graph.nodes()}

    def component_count(self) -> int:
        return len(set(self.labels().values()))

    def recompute(self) -> None:
        """Full rebuild: the honest answer after a removal."""
        self._parent = {}
        self._rank = {}
        for node in self.graph.nodes():
            self._parent[node] = node
            self._rank[node] = 0
        for edge in self.graph.edges():
            self._union(edge.source, edge.target)
        self._stale.clear()
        self.revision += 1

    def _canonical(self, root: str) -> str:
        members = [node for node in self.graph.nodes() if self._find(node) == root]
        return min(members) if members else root

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {"revision": self.revision, "stale": self.is_stale, "nodes": self.graph.node_count}
        if not self.is_stale:
            document["components"] = self.component_count()
        return document
