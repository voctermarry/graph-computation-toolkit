"""Incremental maintenance of weakly connected components and single-source shortest paths.

Edge insertions update the component labelling in near-constant time with a union-find structure.
Edge **removals** cannot be undone that way, so instead of pretending otherwise the structure marks
itself stale and says which algorithms must be recomputed -- an honest incremental API reports what
it no longer knows.

The shortest-path session takes the same stance for *every* successful edge write: negative weights
mean a single insertion can invalidate distances anywhere in the graph, so the snapshot goes stale
and ``distances`` / ``path_to`` refuse until ``recompute`` rebuilds it with Bellman-Ford. The graph's
revision counter distinguishes a real mutation from a mere storage-backend switch, and notices a
mutation made directly on the graph, not only through the session's proxies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .algorithms import bellman_ford
from .errors import NegativeCycleError, ValidationError
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
        self._rank.setdefault(node, 0)
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
        """Insert an edge; returns True when two components merged.

        Insertion never clears staleness. A removal may have split any part of the graph, and a
        single merge -- even re-adding the removed edge -- is not proof the labelling matches the
        whole graph again; only ``recompute`` can confirm that.
        """
        self.graph.add_edge(source, target, weight)
        merged = self._union(source, target)
        self.revision += 1
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


@dataclass(slots=True)
class IncrementalShortestPaths:
    """Bellman-Ford single-source snapshot that refuses rather than answers once the graph moves.

    Construction runs Bellman-Ford, so finite negative weights are allowed and a reachable negative
    cycle raises ``NegativeCycleError`` with the same evidence a direct ``bellman_ford`` call would
    report. Every successful write afterwards -- through the session proxies or directly on the
    graph -- bumps the graph's revision and makes the snapshot stale; a storage-backend switch and a
    write that fails validation bump nothing, so neither does. ``recompute`` is the only way back:
    it rebuilds from the graph as it stands, keeps the previous snapshot if Bellman-Ford still finds
    a reachable negative cycle, and otherwise returns the deterministic distance differential.
    """

    graph: Graph
    source: str
    _distances: dict[str, float] = field(default_factory=dict, init=False)
    _previous: dict[str, str | None] = field(default_factory=dict, init=False)
    # Graph revision the current snapshot was computed at; any mismatch means a mutation happened.
    _snapshot_revision: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        # Delegating to bellman_ford gives the unknown-source ValidationError and the negative-cycle
        # evidence exactly as the standalone algorithm reports them, on every storage backend.
        self._snapshot()

    @property
    def is_stale(self) -> bool:
        return self.graph.revision != self._snapshot_revision

    def _snapshot(self) -> None:
        distance, previous = bellman_ford(self.graph, self.source)
        self._distances = dict(distance)
        self._previous = dict(previous)
        self._snapshot_revision = self.graph.revision

    def _require_fresh(self) -> None:
        if self.is_stale:
            raise ValidationError(
                "shortest paths are stale after a graph mutation; call recompute",
                stale=True,
                source=self.source,
            )

    # -- proxies ---------------------------------------------------------------------------------
    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        """Proxy ``Graph.add_edge``: a successful write makes the snapshot stale, a rejected one nothing."""
        self.graph.add_edge(source, target, weight)

    def remove_edge(self, source: str, target: str) -> bool:
        """Proxy ``Graph.remove_edge``; returns False (and changes nothing) when no such edge exists."""
        return self.graph.remove_edge(source, target)

    # -- answers ---------------------------------------------------------------------------------
    def distances(self) -> dict[str, float]:
        """A copy of the reachable-node distances, or a stale ValidationError after any mutation."""
        self._require_fresh()
        return dict(self._distances)

    def path_to(self, target: str) -> list[str]:
        """A shortest route consistent with the reported distance.

        Fresh state: an unknown node raises ValidationError, a node that exists in the graph but is
        unreachable returns an empty list, and any reachable target returns a real source-to-target
        walk. Stale state raises the stale ValidationError first -- an old path is never presented as
        a current answer.
        """
        self._require_fresh()
        if target not in self.graph.nodes():
            raise ValidationError(f"unknown node: {target}", known=self.graph.nodes()[:10])
        if target not in self._distances:
            return []
        path = [target]
        while self._previous[path[-1]] is not None:
            path.append(str(self._previous[path[-1]]))
        path.reverse()
        return path

    def recompute(self) -> dict[str, object]:
        """Rebuild with Bellman-Ford on the current graph and return the distance differential.

        The differential is measured against the last successfully computed snapshot, so several
        mutations between recomputes produce one combined document instead of a chain. A reachable
        negative cycle propagates NegativeCycleError; the old snapshot and revision pin are kept, so
        the session stays stale and can be recomputed again once the graph is repaired.
        """
        distance, previous = bellman_ford(self.graph, self.source)
        document = self._differential(distance)
        self._distances = dict(distance)
        self._previous = dict(previous)
        self._snapshot_revision = self.graph.revision
        return document

    def _differential(self, new_distances: dict[str, float]) -> dict[str, object]:
        """The source plus added/removed/changed reachable nodes, each section in node-name order."""
        added: dict[str, float] = {}
        removed: dict[str, float] = {}
        changed: dict[str, dict[str, float]] = {}
        for node in sorted(set(self._distances) | set(new_distances)):
            if node in self._distances and node not in new_distances:
                removed[node] = self._distances[node]
            elif node not in self._distances and node in new_distances:
                added[node] = new_distances[node]
            elif new_distances[node] != self._distances[node]:
                changed[node] = {"before": self._distances[node], "after": new_distances[node]}
        return {"source": self.source, "added": added, "removed": removed, "changed": changed}
