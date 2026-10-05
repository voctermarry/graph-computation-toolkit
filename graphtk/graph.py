"""Graph construction and its storage backends.

Nodes are strings. Edges carry a numeric weight (default 1.0). Iteration order is always sorted, so
every algorithm in this package is deterministic without depending on insertion order.

The graph's adjacency can live in three interchangeable backends, selected with the ``storage``
keyword and convertible at any time via :meth:`Graph.to_storage`:

* ``adjacency`` -- the legacy dict of node -> {neighbour: weight};
* ``csr`` -- compressed sparse row: sorted nodes, row offsets, and flat target/weight arrays that
  stay mutable (inserts and deletes shift the arrays);
* ``dense`` -- an n x n weight matrix plus an n x n presence bitmask, so a zero-weight edge is
  still distinguishable from no edge at all.

All three back the same public API -- ``neighbors``, ``edges``, ``degree`` and every algorithm read
through it -- and all three stay mutable: ``add_node``, ``add_edge`` and ``remove_edge`` take effect
immediately regardless of the backend.

Logical memory accounting (used by the ``auto`` selection and reported by ``stats``): with ``n``
nodes and ``a`` adjacency entries (a directed arc counts once, an undirected non-self-loop edge
counts twice -- once per endpoint -- and a self-loop counts once), CSR occupies
``8 * (n + 1) + 16 * a`` bytes and the dense matrix ``9 * n * n`` bytes (an 8-byte weight plus a
1-byte presence flag per cell). ``auto`` picks the smaller of the two, preferring CSR on a tie, on
an empty graph, and whenever the dense footprint would exceed ``DENSE_LIMIT_BYTES``; an explicit
``dense`` request beyond that limit is refused before anything is allocated.
"""

from __future__ import annotations

import bisect
import math
from array import array
from dataclasses import dataclass, field

from .errors import ValidationError

#: Hard ceiling for the dense backend's logical footprint in bytes (512 MiB).
DENSE_LIMIT_BYTES = 536_870_912

#: Storage kinds accepted on the command line; ``auto`` resolves to one of the concrete backends.
STORAGE_KINDS = ("adjacency", "csr", "dense", "auto")

#: The concrete backends a graph can actually live on.
STORAGE_BACKENDS = ("adjacency", "csr", "dense")


def csr_logical_bytes(node_count: int, adjacency_entries: int) -> int:
    """CSR footprint: 8 bytes per offset slot (n + 1 of them) plus 16 per adjacency entry."""
    return 8 * (node_count + 1) + 16 * adjacency_entries


def dense_logical_bytes(node_count: int) -> int:
    """Dense footprint: 9 bytes per matrix cell -- an 8-byte weight and a 1-byte presence flag."""
    return 9 * node_count * node_count


def ensure_dense_allowed(dense_bytes: int) -> None:
    """Refuse a dense allocation beyond the hard limit before any cell is created."""
    if dense_bytes > DENSE_LIMIT_BYTES:
        raise ValidationError(
            "dense storage would exceed the allocation limit",
            requestedBytes=dense_bytes,
            limitBytes=DENSE_LIMIT_BYTES,
        )


def select_storage(requested: str, node_count: int, adjacency_entries: int) -> str:
    """Resolve a requested storage kind to a concrete backend name.

    ``auto`` compares the logical footprints after the graph is fully loaded and takes the smaller
    one; a tie, an empty graph, or a dense footprint over the limit all resolve to ``csr``. An
    explicit ``dense`` request over the limit is a validation error, and so is an unknown kind.
    """
    if requested not in STORAGE_KINDS:
        raise ValidationError(f"unknown storage: {requested}", value=requested, known=list(STORAGE_KINDS))
    if requested == "dense":
        ensure_dense_allowed(dense_logical_bytes(node_count))
        return "dense"
    if requested in STORAGE_BACKENDS:
        return requested
    dense_bytes = dense_logical_bytes(node_count)
    if node_count and dense_bytes < csr_logical_bytes(node_count, adjacency_entries) and dense_bytes <= DENSE_LIMIT_BYTES:
        return "dense"
    return "csr"


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


# -- storage backends ------------------------------------------------------------------------------
# Every backend stores directed adjacency items (the Graph layer mirrors them for undirected
# graphs) and offers the same contract: sorted node order, sorted neighbour order, and immediate
# visibility of mutations.


class _AdjacencyBackend:
    """The legacy storage: a dict of node -> {neighbour: weight}."""

    def __init__(self) -> None:
        self.adjacency: dict[str, dict[str, float]] = {}

    def nodes(self) -> list[str]:
        return sorted(self.adjacency)

    def node_count(self) -> int:
        return len(self.adjacency)

    def has_node(self, node: str) -> bool:
        return node in self.adjacency

    def add_node(self, node: str) -> None:
        self.adjacency.setdefault(node, {})

    def set_edge(self, source: str, target: str, weight: float) -> None:
        self.adjacency[source][target] = weight

    def remove_edge(self, source: str, target: str) -> bool:
        if source in self.adjacency and target in self.adjacency[source]:
            del self.adjacency[source][target]
            return True
        return False

    def has_edge(self, source: str, target: str) -> bool:
        return source in self.adjacency and target in self.adjacency[source]

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        row = self.adjacency[node]
        return [(target, row[target]) for target in sorted(row)]

    def degree(self, node: str) -> int:
        return len(self.adjacency[node])

    def entry_count(self) -> int:
        return sum(len(row) for row in self.adjacency.values())

    def snapshot(self) -> dict[str, dict[str, float]]:
        return self.adjacency


class _CsrBackend:
    """Compressed sparse row that stays mutable: inserts and deletes shift the flat arrays.

    Nodes are kept sorted; each row's targets are kept sorted within the row, so neighbour
    iteration is a slice of the flat arrays in exactly the order the public API promises.
    """

    def __init__(self) -> None:
        self._nodes: list[str] = []
        self._offsets: list[int] = [0]
        self._targets: list[str] = []
        self._weights: list[float] = []

    def _row_index(self, node: str) -> int:
        index = bisect.bisect_left(self._nodes, node)
        if index == len(self._nodes) or self._nodes[index] != node:
            raise KeyError(node)
        return index

    def nodes(self) -> list[str]:
        return list(self._nodes)

    def node_count(self) -> int:
        return len(self._nodes)

    def has_node(self, node: str) -> bool:
        index = bisect.bisect_left(self._nodes, node)
        return index < len(self._nodes) and self._nodes[index] == node

    def add_node(self, node: str) -> None:
        index = bisect.bisect_left(self._nodes, node)
        if index < len(self._nodes) and self._nodes[index] == node:
            return
        self._nodes.insert(index, node)
        self._offsets.insert(index + 1, self._offsets[index])

    def set_edge(self, source: str, target: str, weight: float) -> None:
        index = self._row_index(source)
        start, end = self._offsets[index], self._offsets[index + 1]
        position = bisect.bisect_left(self._targets, target, start, end)
        if position < end and self._targets[position] == target:
            self._weights[position] = weight
            return
        self._targets.insert(position, target)
        self._weights.insert(position, weight)
        for row in range(index + 1, len(self._offsets)):
            self._offsets[row] += 1

    def remove_edge(self, source: str, target: str) -> bool:
        if not self.has_node(source):
            return False
        index = self._row_index(source)
        start, end = self._offsets[index], self._offsets[index + 1]
        position = bisect.bisect_left(self._targets, target, start, end)
        if position == end or self._targets[position] != target:
            return False
        del self._targets[position]
        del self._weights[position]
        for row in range(index + 1, len(self._offsets)):
            self._offsets[row] -= 1
        return True

    def has_edge(self, source: str, target: str) -> bool:
        if not self.has_node(source):
            return False
        index = self._row_index(source)
        start, end = self._offsets[index], self._offsets[index + 1]
        position = bisect.bisect_left(self._targets, target, start, end)
        return position < end and self._targets[position] == target

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        index = self._row_index(node)
        start, end = self._offsets[index], self._offsets[index + 1]
        return list(zip(self._targets[start:end], self._weights[start:end]))

    def degree(self, node: str) -> int:
        index = self._row_index(node)
        return self._offsets[index + 1] - self._offsets[index]

    def entry_count(self) -> int:
        return self._offsets[-1]

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {node: dict(self.neighbors(node)) for node in self._nodes}


class _DenseBackend:
    """Dense n x n storage: a float matrix plus a presence bitmask.

    The bitmask is what keeps a zero-weight edge distinguishable from no edge; nodes stay sorted,
    so scanning a row left to right yields neighbours in sorted order.
    """

    def __init__(self) -> None:
        self._nodes: list[str] = []
        self._index: dict[str, int] = {}
        self._weights: list[array] = []
        self._present: list[bytearray] = []

    def nodes(self) -> list[str]:
        return list(self._nodes)

    def node_count(self) -> int:
        return len(self._nodes)

    def has_node(self, node: str) -> bool:
        return node in self._index

    def add_node(self, node: str) -> None:
        if node in self._index:
            return
        position = bisect.bisect_left(self._nodes, node)
        self._nodes.insert(position, node)
        for shifted in self._nodes[position + 1 :]:
            self._index[shifted] += 1
        self._index[node] = position
        size = len(self._nodes)
        self._weights.insert(position, array("d", [0.0]) * (size - 1))
        self._present.insert(position, bytearray(size - 1))
        for weights_row, present_row in zip(self._weights, self._present):
            weights_row.insert(position, 0.0)
            present_row.insert(position, 0)

    def set_edge(self, source: str, target: str, weight: float) -> None:
        row, column = self._index[source], self._index[target]
        self._weights[row][column] = weight
        self._present[row][column] = 1

    def remove_edge(self, source: str, target: str) -> bool:
        if source not in self._index or target not in self._index:
            return False
        row, column = self._index[source], self._index[target]
        if not self._present[row][column]:
            return False
        self._present[row][column] = 0
        self._weights[row][column] = 0.0
        return True

    def has_edge(self, source: str, target: str) -> bool:
        return source in self._index and target in self._index and bool(self._present[self._index[source]][self._index[target]])

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        row = self._index[node]
        present, weights = self._present[row], self._weights[row]
        return [(self._nodes[column], weights[column]) for column in range(len(self._nodes)) if present[column]]

    def degree(self, node: str) -> int:
        return sum(self._present[self._index[node]])

    def entry_count(self) -> int:
        return sum(sum(row) for row in self._present)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {node: dict(self.neighbors(node)) for node in self._nodes}


def _make_backend(storage: str) -> _AdjacencyBackend | _CsrBackend | _DenseBackend:
    if storage == "adjacency":
        return _AdjacencyBackend()
    if storage == "csr":
        return _CsrBackend()
    if storage == "dense":
        return _DenseBackend()
    raise ValidationError(f"unknown storage: {storage}", value=storage, known=list(STORAGE_BACKENDS))


@dataclass(slots=True)
class Graph:
    directed: bool = False
    storage: str = "adjacency"
    _backend: _AdjacencyBackend | _CsrBackend | _DenseBackend = field(default=None, init=False, repr=False, compare=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._backend = _make_backend(self.storage)

    @property
    def _adjacency(self) -> dict[str, dict[str, float]]:
        """Compatibility view of the legacy layout; the adjacency backend returns its live dict."""
        return self._backend.snapshot()

    # -- construction ----------------------------------------------------------------------------
    def add_node(self, node: str) -> None:
        if not node:
            raise ValidationError("node name must be non-empty")
        self._backend.add_node(node)

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        # Validate before touching any state: a rejected weight must add neither endpoint, overwrite
        # no existing edge, and -- for an undirected graph -- never leave just one half behind.
        value = _validated_weight(weight)
        self.add_node(source)
        self.add_node(target)
        self._backend.set_edge(source, target, value)
        if not self.directed:
            self._backend.set_edge(target, source, value)

    def remove_edge(self, source: str, target: str) -> bool:
        removed = self._backend.remove_edge(source, target)
        if not self.directed:
            removed = self._backend.remove_edge(target, source) or removed
        return removed

    @classmethod
    def from_edges(cls, edges: list[Edge], *, directed: bool = False, storage: str = "adjacency") -> "Graph":
        graph = cls(directed=directed, storage=storage)
        for edge in edges:
            graph.add_edge(edge.source, edge.target, edge.weight)
        return graph

    def to_storage(self, storage: str) -> "Graph":
        """A copy of this graph on another storage backend; every node, edge and weight survives."""
        if storage == "dense":
            ensure_dense_allowed(dense_logical_bytes(self.node_count))
        clone = Graph(directed=self.directed, storage=storage)
        for node in self.nodes():
            clone.add_node(node)
        for edge in self.edges():
            clone.add_edge(edge.source, edge.target, edge.weight)
        return clone

    # -- queries ---------------------------------------------------------------------------------
    @property
    def node_count(self) -> int:
        return self._backend.node_count()

    def nodes(self) -> list[str]:
        return self._backend.nodes()

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        if not self._backend.has_node(node):
            raise ValidationError(f"unknown node: {node}", known=self.nodes()[:10])
        return self._backend.neighbors(node)

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
        if not self._backend.has_node(node):
            raise ValidationError(f"unknown node: {node}")
        return self._backend.degree(node)

    def weighted_degree(self, node: str) -> float:
        return sum(weight for _, weight in self.neighbors(node))

    def adjacency_entry_count(self) -> int:
        """Total adjacency items: arcs once, undirected non-self-loop edges twice, self-loops once."""
        return self._backend.entry_count()

    def self_loops(self) -> list[str]:
        return [node for node in self.nodes() if self._backend.has_edge(node, node)]

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
