"""Graph construction, the CSR representation, and selectable read backends.

Nodes are strings. Edges carry a numeric weight (default 1.0). Iteration order is always sorted, so
every algorithm in this package is deterministic without depending on insertion order.

The mutable adjacency dict stays the single write layer -- ``add_node``, ``add_edge`` and
``remove_edge`` keep exactly their old semantics (undirected mirroring, self-loops, last-write-wins
on duplicates). ``configure_storage`` adds an optional read layer that ``nodes``, ``neighbors``,
``edges``, ``degree`` and every algorithm consult instead:

* ``csr``   -- compressed sparse row: an 8-byte offset per node boundary plus a flattened
               16-byte (node id, weight) neighbour list, both kept sorted by node name;
* ``dense`` -- an n*n presence byte plus an n*n float64 weight, 9 bytes per cell. The presence byte
               is what lets a zero-weight edge stay distinct from a pair that has no edge;
* ``auto``  -- the smaller of the two logical footprints, CSR winning ties and every empty graph.

A revision counter invalidates the arrays: any mutation rebuilds lazily on the next query, so a
switched graph still supports ``add_node``/``add_edge``/``remove_edge`` and immediately reflects them.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field

from .errors import ValidationError

DENSE_LIMIT_BYTES = 536870912
STORAGE_CHOICES = ("adjacency", "csr", "dense", "auto")


def _validated_weight(weight: object) -> float:
    """Normalise an edge weight to float, or raise ValidationError.

    ``bool`` is rejected even though it subclasses ``int``; strings and ``None`` are rejected even
    though ``float()`` would accept some of them; and NaN plus the two infinities are rejected
    because a single non-finite value poisons every distance, rank and JSON document it reaches.
    Negative finite values are legal -- Bellman-Ford needs them.
    """
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValidationError("edge weight must be a finite number", value=weight)
    value = float(weight)
    if not math.isfinite(value):
        raise ValidationError("edge weight must be finite")
    return value


@dataclass(frozen=True, slots=True)
class Edge:
    source: str
    target: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValidationError("edge endpoints must be non-empty", value=f"{self.source!r}->{self.target!r}")
        object.__setattr__(self, "weight", _validated_weight(self.weight))

    def to_document(self) -> dict[str, object]:
        return {"source": self.source, "target": self.target, "weight": float(self.weight)}


@dataclass(slots=True)
class Graph:
    directed: bool = False
    _adjacency: dict[str, dict[str, float]] = field(default_factory=dict, init=False)

    # Backend actually read on queries: "adjacency" (the default), "csr" or "dense". ``auto`` is
    # resolved at selection time, so this field always names the materialised backend.
    _storage: str = field(default="adjacency", init=False)
    # What the caller passed to configure_storage; "auto" is preserved here for the stats report.
    _storage_requested: str | None = field(default=None, init=False)
    # Bumped on every mutation; the arrays carry the revision they were built at and rebuild
    # lazily whenever it lags (a same-shape weight overwrite must invalidate too).
    _revision: int = field(default=0, init=False)
    _built_revision: int = field(default=-1, init=False)
    # CSR arrays.
    _csr_nodes: list[str] = field(default_factory=list, init=False)
    _csr_offsets: array = field(default_factory=lambda: array("q"), init=False)
    _csr_columns: array = field(default_factory=lambda: array("q"), init=False)
    _csr_weights: array = field(default_factory=lambda: array("d"), init=False)
    # Dense arrays. _dense_present[row * n + col] distinguishes a stored edge (its weight may be 0)
    # from a pair with no edge at all.
    _dense_nodes: list[str] = field(default_factory=list, init=False)
    _dense_index: dict[str, int] = field(default_factory=dict, init=False)
    _dense_present: bytearray = field(default_factory=bytearray, init=False)
    _dense_weights: array = field(default_factory=lambda: array("d"), init=False)

    # -- construction ----------------------------------------------------------------------------
    def add_node(self, node: str) -> None:
        if not node:
            raise ValidationError("node name must be non-empty")
        if node not in self._adjacency:
            self._adjacency[node] = {}
            self._revision += 1

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        # Validate before touching any state: a rejected weight must add neither endpoint, overwrite
        # no existing edge, and -- for an undirected graph -- never leave just one half behind.
        value = _validated_weight(weight)
        self.add_node(source)
        self.add_node(target)
        if self.directed:
            self._adjacency[source][target] = value
        else:
            self._adjacency[source][target] = value
            self._adjacency[target][source] = value
        # A write of adjacency -- new arc or last-write-wins overwrite -- invalidates the arrays.
        self._revision += 1

    def remove_edge(self, source: str, target: str) -> bool:
        removed = False
        if source in self._adjacency and target in self._adjacency[source]:
            del self._adjacency[source][target]
            removed = True
        if not self.directed and target in self._adjacency and source in self._adjacency[target]:
            del self._adjacency[target][source]
            removed = True
        if removed:
            self._revision += 1
        return removed

    @classmethod
    def from_edges(cls, edges: list[Edge], *, directed: bool = False) -> "Graph":
        graph = cls(directed=directed)
        for edge in edges:
            graph.add_edge(edge.source, edge.target, edge.weight)
        return graph

    # -- storage selection -----------------------------------------------------------------------
    def storage_sizes(self) -> tuple[int, int, int]:
        """Logical footprints ``(adjacency_entries, csr_bytes, dense_bytes)``.

        ``n`` is the node count and ``a`` the total number of adjacency entries across every node:
        each directed arc counts once, an undirected non-self-loop edge counts twice (reachable from
        both ends) and a self-loop once. CSR is ``8*(n+1) + 16*a`` bytes; dense is ``9*n*n``.
        """
        n = len(self._adjacency)
        entries = sum(len(neighbours) for neighbours in self._adjacency.values())
        csr_bytes = 8 * (n + 1) + 16 * entries
        dense_bytes = 9 * n * n
        return entries, csr_bytes, dense_bytes

    def _build_csr(self) -> None:
        nodes = sorted(self._adjacency)
        index = {node: position for position, node in enumerate(nodes)}
        offsets = array("q", [0])
        columns = array("q")
        weights = array("d")
        for node in nodes:
            for target in sorted(self._adjacency[node]):
                columns.append(index[target])
                weights.append(self._adjacency[node][target])
            offsets.append(len(columns))
        self._csr_nodes, self._csr_offsets = nodes, offsets
        self._csr_columns, self._csr_weights = columns, weights

    def _build_dense(self) -> None:
        nodes = sorted(self._adjacency)
        n = len(nodes)
        required = 9 * n * n
        # The limit holds for the lazy post-mutation rebuild too: a graph that grew past it must
        # never attempt the allocation on its next query.
        if required > DENSE_LIMIT_BYTES:
            raise ValidationError(
                "dense storage exceeds the allocation limit",
                requestedBytes=required,
                limitBytes=DENSE_LIMIT_BYTES,
            )
        index = {node: position for position, node in enumerate(nodes)}
        present = bytearray(n * n)
        weights = array("d", [0.0]) * (n * n)
        for source in nodes:
            row = index[source] * n
            for target, weight in self._adjacency[source].items():
                cell = row + index[target]
                present[cell] = 1
                weights[cell] = weight
        self._dense_nodes, self._dense_index = nodes, index
        self._dense_present, self._dense_weights = present, weights

    def configure_storage(self, requested: str = "adjacency") -> dict[str, object]:
        """Select the backend used by every read, and return the stable stats ``storage`` document.

        ``adjacency`` restores the mutable dict. ``csr`` and ``dense`` materialise their arrays now
        (and re-materialise lazily after any mutation). ``auto`` compares the logical footprints
        after a full load and keeps the smaller one -- CSR on a tie, and always CSR for an empty
        graph; an oversized dense candidate under ``auto`` also falls back to CSR. An explicit
        ``dense`` whose matrix would exceed ``DENSE_LIMIT_BYTES`` is refused before any allocation.
        """
        if requested not in STORAGE_CHOICES:
            raise ValidationError(
                f"storage must be one of {', '.join(STORAGE_CHOICES)}",
                value=requested,
            )
        entries, csr_bytes, dense_bytes = self.storage_sizes()
        n = len(self._adjacency)
        selected = requested
        if requested == "auto":
            selected = "csr" if n == 0 or csr_bytes <= dense_bytes else "dense"
            if selected == "dense" and dense_bytes > DENSE_LIMIT_BYTES:
                selected = "csr"
        if selected == "dense":
            if dense_bytes > DENSE_LIMIT_BYTES:
                raise ValidationError(
                    "dense storage exceeds the allocation limit",
                    requestedBytes=dense_bytes,
                    limitBytes=DENSE_LIMIT_BYTES,
                )
            self._build_dense()
        elif selected == "csr":
            self._build_csr()
        self._storage = selected
        self._storage_requested = requested
        self._built_revision = self._revision
        return self.storage_document()

    def _ensure_storage(self) -> None:
        """Rebuild the active arrays if a mutation happened after they were built."""
        if self._built_revision == self._revision:
            return
        if self._storage == "csr":
            self._build_csr()
        elif self._storage == "dense":
            self._build_dense()
        self._built_revision = self._revision

    def storage_document(self) -> dict[str, object]:
        """The ``storage`` object stats emits: request, selection, footprints and density."""
        entries, csr_bytes, dense_bytes = self.storage_sizes()
        n = len(self._adjacency)
        density = round(entries / (n * n), 10) if n else 0
        # adjacency has no formula of its own: its logical footprint is the same adjacency entries
        # laid out as CSR, which is the encoding csr() already exports.
        logical_bytes = {"adjacency": csr_bytes, "csr": csr_bytes, "dense": dense_bytes}[self._storage]
        return {
            "requested": self._storage_requested or "adjacency",
            "selected": self._storage,
            "logicalBytes": logical_bytes,
            "csrBytes": csr_bytes,
            "denseBytes": dense_bytes,
            "adjacencyEntries": entries,
            "density": density,
        }

    # -- queries ---------------------------------------------------------------------------------
    @property
    def node_count(self) -> int:
        return len(self._adjacency)

    @property
    def revision(self) -> int:
        """The mutation counter: bumped by every successful node/edge write, never by a storage switch.

        Incremental sessions pin their snapshot to a revision, so a direct ``add_node`` /
        ``add_edge`` / ``remove_edge`` on this graph is visible to them without any notification.
        """
        return self._revision

    def nodes(self) -> list[str]:
        if self._storage == "csr":
            self._ensure_storage()
            return list(self._csr_nodes)
        if self._storage == "dense":
            self._ensure_storage()
            return list(self._dense_nodes)
        return sorted(self._adjacency)

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        if self._storage == "csr":
            self._ensure_storage()
            try:
                row = self._csr_nodes.index(node)
            except ValueError:
                raise ValidationError(f"unknown node: {node}", known=list(self._csr_nodes[:10])) from None
            start, end = self._csr_offsets[row], self._csr_offsets[row + 1]
            return [
                (self._csr_nodes[self._csr_columns[position]], self._csr_weights[position])
                for position in range(start, end)
            ]
        if self._storage == "dense":
            self._ensure_storage()
            if node not in self._dense_index:
                raise ValidationError(f"unknown node: {node}", known=list(self._dense_nodes[:10]))
            n = len(self._dense_nodes)
            row = self._dense_index[node] * n
            return [
                (self._dense_nodes[column], self._dense_weights[row + column])
                for column in range(n)
                if self._dense_present[row + column]
            ]
        if node not in self._adjacency:
            raise ValidationError(f"unknown node: {node}", known=self.nodes()[:10])
        return [(target, self._adjacency[node][target]) for target in sorted(self._adjacency[node])]

    def edges(self) -> list[Edge]:
        seen: set[tuple[str, str]] = set()
        result: list[Edge] = []
        for source in self.nodes():
            for target, weight in self.neighbors(source):
                if not self.directed and (target, source) in seen:
                    continue
                seen.add((source, target))
                result.append(Edge(source, target, weight))
        return result

    def edge_count(self) -> int:
        return len(self.edges())

    def degree(self, node: str) -> int:
        return len(self.neighbors(node))

    def weighted_degree(self, node: str) -> float:
        # Adjacency keeps the historical summation order (dict values) exactly; the array backends
        # read their own weights. Float addition is not associative, so the order must not drift.
        self.neighbors(node)
        if self._storage == "adjacency":
            return sum(self._adjacency[node].values())
        return sum(weight for _, weight in self.neighbors(node))

    def self_loops(self) -> list[str]:
        return [node for node in self.nodes() if any(neighbour == node for neighbour, _ in self.neighbors(node))]

    def has_negative_weights(self) -> bool:
        return any(edge.weight < 0 for edge in self.edges())

    def csr(self) -> tuple[list[str], list[int], list[tuple[str, float]]]:
        """Compressed sparse row: node order, offsets, and the flattened neighbour list."""
        if self._storage == "csr":
            self._ensure_storage()
            nodes = list(self._csr_nodes)
            offsets = list(self._csr_offsets)
            flat = [
                (nodes[self._csr_columns[position]], self._csr_weights[position])
                for position in range(len(self._csr_columns))
            ]
            return nodes, offsets, flat
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
