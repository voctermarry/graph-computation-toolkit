"""Incremental maintenance of weakly connected components and single-source shortest paths.

Edge insertions update the component labelling in near-constant time with a union-find structure.
Edge **removals** cannot be undone that way, so instead of pretending otherwise the structure marks
itself stale and says which algorithms must be recomputed -- an honest incremental API reports what
it no longer knows.

The shortest-path session takes the same stance one level up: it keeps the last successfully
computed Bellman-Ford snapshot, watches the graph's revision counter for any structural or weight
change (however it was made), refuses to serve outdated answers while stale, and reports exactly
what changed when it recomputes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .algorithms import bellman_ford, path_from
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
    """Single-source shortest paths that track a mutable graph, with honest staleness.

    The session snapshots Bellman-Ford at construction: negative edges are legal, an unknown source
    is a ``ValidationError``, and a reachable negative cycle propagates ``NegativeCycleError`` with
    its evidence. Afterwards the graph's revision counter is the change detector -- it moves on
    every successful structural or weight write, whether the write went through this session's
    ``add_edge``/``remove_edge`` proxies or straight through the ``Graph``, and it does not move for
    a storage-backend switch, a removal that found nothing, or a rejected (invalid) write. Any real
    change makes the snapshot stale: ``distances`` and ``path_to`` then refuse with a
    ``ValidationError`` carrying ``stale: True`` rather than serve an outdated answer, and
    ``recompute`` is the only way back. Recomputation always diffs against the last *successful*
    snapshot, and a recomputation that meets a negative cycle keeps that snapshot untouched -- the
    session stays stale until the graph is fixed and a recompute succeeds.
    """

    graph: Graph
    source: str
    _distance: dict[str, float] = field(default_factory=dict, init=False)
    _previous: dict[str, str | None] = field(default_factory=dict, init=False)
    # The graph revision the snapshot was computed at; anything else means the graph moved on.
    _snapshot_revision: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        distance, previous = bellman_ford(self.graph, self.source)
        self._distance = distance
        self._previous = previous
        self._snapshot_revision = self.graph._revision

    # -- maintenance -----------------------------------------------------------------------------
    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        """Insert or overwrite an edge through the graph; a rejected write changes nothing."""
        self.graph.add_edge(source, target, weight)

    def remove_edge(self, source: str, target: str) -> bool:
        """Remove an edge through the graph; ``False`` (nothing was there) leaves the session fresh."""
        return self.graph.remove_edge(source, target)

    @property
    def is_stale(self) -> bool:
        return self.graph._revision != self._snapshot_revision

    def _require_fresh(self) -> None:
        if self.is_stale:
            raise ValidationError(
                "shortest paths are stale after a graph change; call recompute",
                stale=True,
            )

    # -- answers ---------------------------------------------------------------------------------
    @property
    def distances(self) -> dict[str, float]:
        """A copy of the reachable distance map; unreachable nodes are absent, never faked."""
        self._require_fresh()
        return dict(self._distance)

    def path_to(self, target: str) -> list[str]:
        """The snapshot's path to `target`: a real route, ``[]`` when unreachable, an error when unknown."""
        self._require_fresh()
        if target not in self.graph.nodes():
            raise ValidationError(f"unknown node: {target}", known=self.graph.nodes()[:10])
        if target not in self._previous:
            return []
        return path_from(self._previous, target)

    def recompute(self) -> dict[str, object]:
        """Re-run Bellman-Ford on the current graph and report the diff against the last snapshot.

        On success the session is fresh again and the returned document names the source plus three
        sorted maps: ``added`` (newly reachable node -> distance), ``removed`` (newly unreachable
        node -> its old distance) and ``changed`` (node -> ``before``/``after``); nodes whose
        distance survived are not mentioned. A reachable negative cycle raises
        ``NegativeCycleError`` instead: the stale flag and the previous snapshot survive, so a
        repaired graph can simply be recomputed again.
        """
        distance, previous = bellman_ford(self.graph, self.source)
        old = self._distance
        added = {node: distance[node] for node in sorted(distance) if node not in old}
        removed = {node: old[node] for node in sorted(old) if node not in distance}
        changed = {
            node: {"before": old[node], "after": distance[node]}
            for node in sorted(old)
            if node in distance and old[node] != distance[node]
        }
        self._distance = distance
        self._previous = previous
        self._snapshot_revision = self.graph._revision
        return {"source": self.source, "added": added, "removed": removed, "changed": changed}
