"""Reproducible state-sequence differential tests across the storage backends.

The adjacency backend is the isomorphic reference graph and never leaves it. For each graph type
(directed and undirected) one reference graph and three peer graphs -- ``csr``, ``dense`` and
``auto``, configured to their named backend before the replay begins -- replay the *exact same*
sequence of ``add_node`` / ``add_edge`` / ``remove_edge`` / ``configure_storage`` operations. After
every successful step the peers must match the reference on the whole public observation surface
(nodes, edges, neighbours, degree, edge_count, self_loops, csr, to_document), on the storage sizes
and selection report (each interpreted for the backend currently selected), and -- whenever the
state's preconditions allow it -- on the complete results of every algorithm: BFS, DFS, Dijkstra,
Bellman-Ford, components, degree centrality, clustering, PageRank and label propagation. Directed
states additionally compare the topological sort (a cyclic state compares the raised CycleError
instead), reachable targets are compared through A* with the all-zero heuristic, and every
shortest-path ``path_from`` reconstruction is re-walked over edges that must still exist with weights
summing to the reported distance.

The sequences come from a fixed linear congruential generator seeded with hard-coded values, so
repeated runs always choose the same cases and reach the same verdict -- no third-party random
testing library and no wall-clock input. Every failure message names the graph type, the backend,
the seed and the index of the first divergent step.

The hand-written anchor sequences guarantee the required situations are actually visited in order:
isolated nodes, self-loops, zero-weight edges, duplicate-edge weight overwrites, removing a
non-existent edge (still ``False``), mutation after a backend switch, same-shape weight overwrites,
and consecutive switches between the materialised backends. Invalid operations (empty node names,
non-finite or non-numeric weights) must keep raising ``ValidationError`` on every backend with
identical evidence and leave no trace: nodes, edges, storage selection and subsequent algorithm
results are all unchanged.
"""

from __future__ import annotations

import unittest
from collections import Counter

from graphtk import (
    Graph,
    CycleError,
    NegativeCycleError,
    NegativeWeightError,
    ValidationError,
    astar,
    bellman_ford,
    bfs,
    clustering,
    components,
    degree_centrality,
    dfs,
    dijkstra,
    label_propagation,
    path_from,
    pagerank,
    topological_sort,
)

PEERS = ("csr", "dense", "auto")
ALL_BACKENDS = ("adjacency", "csr", "dense", "auto")

# Small finite weight universe: integers (exact in float64) plus the zero-weight case and a few
# dyadic fractions, so float sums compare exactly across summation orders. A legal negative value
# is included, so Dijkstra/A*/PageRank/label-propagation refusal paths occur as ordinary sequence
# steps; NaN/inf/strings/bool only appear as deliberately invalid operations.
WEIGHTS = [-3, -1, 0, 1, 2, 3, 5, 7, 0.5, 1.25, 2.5, 4.0]
INVALID_WEIGHTS = [float("nan"), float("inf"), float("-inf"), "2.0", None, True]
INVALID_NODE_NAMES = [""]  # add_node/add_edge reject exactly the empty string

# Fixed LCG (Numerical Recipes constants), modulus 2**32: every "random" choice this module makes
# derives from it, so a given seed always reproduces the identical operation sequence.
LCG_MODULUS = 1 << 32
LCG_MULTIPLIER = 1664525
LCG_INCREMENT = 1013904223

GENERATED_SEEDS = (1, 2, 3, 4, 5, 6)
GENERATED_STEPS = 64
NODE_POOL = 6  # n <= 6 keeps dense far below the allocation limit and every algorithm cheap.


# -- deterministic sequence source ----------------------------------------------------------------
class SequenceRng:
    """The only source of choices: a seeded LCG with small bounded draws."""

    def __init__(self, seed: int) -> None:
        self.state = seed & (LCG_MODULUS - 1)

    def draw(self, bound: int) -> int:
        self.state = (LCG_MULTIPLIER * self.state + LCG_INCREMENT) % LCG_MODULUS
        return self.state % bound

    def pick(self, values):
        return values[self.draw(len(values))]


# The anchor prefix makes the required special cases occur unconditionally, in a known order,
# before the generated churn begins: isolated node, self-loop, zero weight, duplicate overwrite
# (twice, including a same-shape overwrite back to the earlier weight), removal of an absent pair,
# a real deletion, then another self-loop and a re-add.
ANCHOR_PREFIX: tuple[tuple, ...] = (
    ("add_node", "iso"),                            # isolated node
    ("add_edge", "n0", "n0", 1.0),                 # self-loop
    ("add_edge", "n0", "n1", 0.0),                 # zero-weight edge
    ("add_edge", "n1", "n2", 2),                   # ordinary edge
    ("add_edge", "n0", "n1", 3),                   # duplicate edge: weight overwrite
    ("add_edge", "n0", "n1", 0.0),                 # same pair, overwrite back
    ("remove_edge", "n3", "n4"),                   # edge that never existed -> False
    ("remove_edge", "n1", "n2"),                   # real deletion (mirror half if undirected)
    ("add_edge", "n2", "n2", 5),                   # second self-loop
    ("add_edge", "n0", "n1", 1),                   # re-add after the churn
)

# Hand-written tails: explicit post-switch mutation and consecutive materialised switches.
ANCHOR_TAILS: dict[str, list[tuple]] = {
    "switch_then_mutate": [
        ("configure", "csr"),
        ("add_node", "fresh"),
        ("add_edge", "n0", "fresh", 4),
        ("add_edge", "n0", "n1", 7),                # overwrite while on csr
        ("remove_edge", "n0", "fresh"),
        ("configure", "dense"),
        ("add_edge", "fresh", "n2", 2),             # mutate while on dense
        ("remove_edge", "n2", "n2"),                # drop the self-loop
        ("configure", "adjacency"),
        ("add_edge", "iso", "n2", 1),               # mutate back on adjacency
    ],
    "consecutive_materialised_switches": [
        ("configure", "dense"),
        ("configure", "csr"),
        ("configure", "dense"),                     # dense -> csr -> dense, nothing in between
        ("configure", "auto"),
        ("add_edge", "n1", "n3", 0),                # zero-weight after repeated switches
        ("configure", "csr"),
        ("add_edge", "n3", "n3", 9),                # self-loop after repeated switches
        ("configure", "dense"),
        ("remove_edge", "n1", "n3"),
        ("configure", "auto"),
        ("configure", "adjacency"),
    ],
}


def generated_sequence(seed: int) -> list[tuple]:
    """Build one reproducible operation list from a fixed seed.

    The mix is weighted toward edges and removals so the graph churns (overwrites, deletions of
    present and absent pairs, orphaned nodes) while staying tiny. A storage switch lands roughly
    every eighth step, cycling through every selection including adjacency. The anchor prefix is
    prepended and two materialised switches appended, so every generated run shares the required
    shapes and ends on the switched read paths.
    """
    rng = SequenceRng(seed)
    operations: list[tuple] = []
    for step in range(GENERATED_STEPS):
        kind = rng.draw(16)
        if step % 8 == 3:
            operations.append(("configure", rng.pick(ALL_BACKENDS)))
        elif kind < 2:
            operations.append(("add_node", f"n{rng.draw(NODE_POOL)}"))
        elif kind < 12:
            operations.append(
                (
                    "add_edge",
                    f"n{rng.draw(NODE_POOL)}",
                    f"n{rng.draw(NODE_POOL)}",
                    rng.pick(WEIGHTS),
                )
            )
        else:
            operations.append(
                (
                    "remove_edge",
                    f"n{rng.draw(NODE_POOL)}",
                    f"n{rng.draw(NODE_POOL)}",
                )
            )
    return [*ANCHOR_PREFIX, *operations, ("configure", "dense"), ("configure", "csr")]


# -- observation snapshots ------------------------------------------------------------------------
def edge_rows(graph: Graph) -> list[tuple]:
    return [(edge.source, edge.target, edge.weight) for edge in graph.edges()]


def read_snapshot(graph: Graph) -> dict:
    """The public read surface that must be identical on every backend in the same state."""
    nodes = graph.nodes()
    return {
        "nodes": nodes,
        "edges": edge_rows(graph),
        "neighbors": {node: graph.neighbors(node) for node in nodes},
        "degrees": {node: graph.degree(node) for node in nodes},
        "edgeCount": graph.edge_count(),
        "selfLoops": graph.self_loops(),
        "csr": graph.csr(),
        "toDocument": graph.to_document(),
    }


def traversable_weights(graph: Graph) -> dict[tuple[str, str], float]:
    return {
        (node, neighbour): weight
        for node in graph.nodes()
        for neighbour, weight in graph.neighbors(node)
    }


def shortest_path_results(graph: Graph) -> dict:
    """BFS/DFS/Dijkstra/Bellman-Ford/A* for every source (and every reachable target).

    Raises are part of the result: with a negative edge Dijkstra and A* refuse with
    NegativeWeightError while Bellman-Ford answers; a reachable negative cycle makes Bellman-Ford
    raise NegativeCycleError. The refusal documents are captured, so peers agree on exception type
    and evidence, not just on the successful answers. A* always runs in the non-negative case with
    the all-zero heuristic; its refusal document (same evidence for any target, since the edge scan
    happens first) is captured once per source in the negative case.
    """
    nodes = graph.nodes()
    zero_heuristic = {node: 0 for node in nodes}
    results: dict = {"sources": {}}
    for source in nodes:
        entry: dict = {"bfs": bfs(graph, source), "dfs": dfs(graph, source)}
        try:
            distance, previous = dijkstra(graph, source)
            entry["dijkstra"] = ("ok", distance, previous)
        except NegativeWeightError as error:
            entry["dijkstra"] = ("negative_weight", error.to_document())

        try:
            distance, previous = bellman_ford(graph, source)
            entry["bellman"] = ("ok", distance, previous)
        except NegativeCycleError as error:
            entry["bellman"] = ("negative_cycle", error.to_document())

        if entry["dijkstra"][0] == "ok":
            targets: dict[str, tuple] = {}
            for target in entry["dijkstra"][1]:
                targets[target] = astar(graph, source, target, zero_heuristic)
            entry["astar"] = ("ok", targets)
        else:
            # The up-front negative-edge scan refuses before the target is even consulted, so the
            # captured evidence is independent of which target was named.
            try:
                astar(graph, source, nodes[-1], zero_heuristic)
            except NegativeWeightError as error:
                entry["astar"] = ("negative_weight", error.to_document())
            else:  # pragma: no cover - a negative edge makes this branch unreachable
                raise AssertionError("astar accepted a graph carrying a negative edge")
        results["sources"][source] = entry
    return results


def whole_graph_results(graph: Graph) -> dict:
    """Source-free algorithms; PageRank/propagation may legitimately refuse negative edges."""
    results: dict = {
        "components": components(graph),
        "centrality": degree_centrality(graph),
        "clustering": clustering(graph).to_document(),
    }
    try:
        results["pagerank"] = ("ok", pagerank(graph).to_document())
    except NegativeWeightError as error:
        results["pagerank"] = ("negative_weight", error.to_document())
    try:
        results["communities"] = ("ok", label_propagation(graph).to_document())
    except NegativeWeightError as error:
        results["communities"] = ("negative_weight", error.to_document())
    if graph.directed:
        try:
            results["toposort"] = ("ok", topological_sort(graph))
        except CycleError as error:
            results["toposort"] = ("cycle", error.to_document())
    else:
        # Undirected graphs never offer a topological order; the refusal itself must match.
        try:
            topological_sort(graph)
        except ValidationError as error:
            results["toposort"] = ("validation", error.to_document())
        else:  # pragma: no cover - the production contract forbids this branch
            raise AssertionError("topological_sort unexpectedly accepted an undirected graph")
    return results


def reconstructed_paths(shortest: dict) -> list[tuple]:
    """Every path_from reconstruction as comparable plain data.

    Each item is (source, target, algorithm, path, distance); the re-walk assertion then checks the
    path against the graph it came from.
    """
    paths: list[tuple] = []
    for source, entry in shortest["sources"].items():
        predecessor_sets: list[tuple[str, dict, dict]] = []
        if entry["dijkstra"][0] == "ok":
            predecessor_sets.append(("dijkstra", entry["dijkstra"][1], entry["dijkstra"][2]))
        if entry["bellman"][0] == "ok":
            predecessor_sets.append(("bellman", entry["bellman"][1], entry["bellman"][2]))
        for label, distances, previous in predecessor_sets:
            for target, distance in distances.items():
                paths.append((source, target, label, tuple(path_from(previous, target)), distance))
        if entry["astar"][0] == "ok":
            for target, (distance, previous, _expanded) in entry["astar"][1].items():
                paths.append((source, target, "astar", tuple(path_from(previous, target)), distance))
    return paths


def assert_path_is_real(
    case: unittest.TestCase,
    graph: Graph,
    path_item: tuple,
    context: str,
) -> None:
    source, target, _label, path, distance = path_item
    weights = traversable_weights(graph)
    case.assertEqual(path[0], source, context)
    case.assertEqual(path[-1], target, context)
    total = 0.0
    for left, right in zip(path, path[1:]):
        case.assertIn((left, right), weights, f"{context}: path {path} uses a non-existent edge")
        total += weights[(left, right)]
    case.assertEqual(total, distance, f"{context}: path {path} sums to {total}, not {distance}")


# -- operation application ------------------------------------------------------------------------
def apply_mutating(graph: Graph, operation: tuple):
    """Apply add_node/add_edge/remove_edge; return the call's value (None or a bool)."""
    kind = operation[0]
    if kind == "add_node":
        return graph.add_node(operation[1])
    if kind == "add_edge":
        return graph.add_edge(operation[1], operation[2], operation[3])
    if kind == "remove_edge":
        return graph.remove_edge(operation[1], operation[2])
    raise AssertionError(f"unknown operation {kind!r}")


# -- the differential harness ---------------------------------------------------------------------
class DifferentialHarness:
    def __init__(self, case: unittest.TestCase, directed: bool, seed: object) -> None:
        self.case = case
        self.directed = directed
        self.seed = seed
        self.graph_type = "directed" if directed else "undirected"
        self.reference = Graph(directed=directed)
        self.peers = {name: Graph(directed=directed) for name in PEERS}
        # Peers start already switched, so the anchor prefix exercises lazy rebuilds on each
        # materialised backend; ``auto`` on an empty graph resolves to csr.
        self.requested = {name: name for name in PEERS}
        self.selected = {"csr": "csr", "dense": "dense", "auto": "csr"}
        for name, peer in self.peers.items():
            peer.configure_storage(name)

    def label(self, backend: str, step: int, what: str) -> str:
        return (
            f"graph={self.graph_type}, backend={backend}, seed={self.seed!r}, "
            f"first mismatch at step {step} ({what})"
        )

    def expected_storage_document(self, requested: str, selected: str) -> dict:
        """The storage stats document a graph in the reference's state must report."""
        entries, csr_bytes, dense_bytes = self.reference.storage_sizes()
        n = self.reference.node_count
        density = round(entries / (n * n), 10) if n else 0
        logical_bytes = dense_bytes if selected == "dense" else csr_bytes
        return {
            "requested": requested,
            "selected": selected,
            "logicalBytes": logical_bytes,
            "csrBytes": csr_bytes,
            "denseBytes": dense_bytes,
            "adjacencyEntries": entries,
            "density": density,
        }

    def resolve_auto(self) -> str:
        _entries, csr_bytes, dense_bytes = self.reference.storage_sizes()
        n = self.reference.node_count
        # The dense-limit fallback cannot trigger at these sizes; re-derive the plain rule.
        return "csr" if n == 0 or csr_bytes <= dense_bytes else "dense"

    def check_storage(self, step: int) -> None:
        sizes = self.reference.storage_sizes()
        self.case.assertEqual(
            self.reference.storage_document(),
            self.expected_storage_document("adjacency", "adjacency"),
            self.label("adjacency", step, "storage report"),
        )
        for name, peer in self.peers.items():
            self.case.assertEqual(
                peer.storage_sizes(),
                sizes,
                self.label(name, step, "storage sizes"),
            )
            self.case.assertEqual(
                peer.storage_document(),
                self.expected_storage_document(self.requested[name], self.selected[name]),
                self.label(name, step, "storage report"),
            )

    def check_full_state(self, step: int) -> None:
        expected_reads = read_snapshot(self.reference)
        for name, peer in self.peers.items():
            self.case.assertEqual(
                read_snapshot(peer), expected_reads, self.label(name, step, "public reads")
            )
        self.check_storage(step)
        self.check_algorithms(step)

    def check_algorithms(self, step: int) -> None:
        if not self.reference.nodes():
            return
        expected_shortest = shortest_path_results(self.reference)
        expected_whole = whole_graph_results(self.reference)
        expected_paths = reconstructed_paths(expected_shortest)
        for name, peer in self.peers.items():
            peer_shortest = shortest_path_results(peer)
            self.case.assertEqual(
                peer_shortest, expected_shortest, self.label(name, step, "shortest-path results")
            )
            self.case.assertEqual(
                whole_graph_results(peer), expected_whole, self.label(name, step, "graph results")
            )
            self.case.assertEqual(
                reconstructed_paths(peer_shortest),
                expected_paths,
                self.label(name, step, "path_from reconstructions"),
            )
            for path_item in expected_paths:
                assert_path_is_real(
                    self.case,
                    peer,
                    path_item,
                    self.label(name, step, f"re-walk via {path_item[2]}"),
                )

    def replay(self, operations: list[tuple]) -> None:
        for step, operation in enumerate(operations):
            if operation[0] == "configure":
                requested = operation[1]
                reference_document = self.reference.configure_storage("adjacency")
                self.case.assertEqual(
                    reference_document,
                    self.expected_storage_document("adjacency", "adjacency"),
                    self.label("adjacency", step, "configure_storage result"),
                )
                for name, peer in self.peers.items():
                    document = peer.configure_storage(requested)
                    resolved = requested if requested in ("csr", "dense", "adjacency") else self.resolve_auto()
                    self.case.assertEqual(
                        document,
                        self.expected_storage_document(requested, resolved),
                        self.label(name, step, "configure_storage result"),
                    )
                    self.requested[name] = requested
                    self.selected[name] = resolved
            else:
                expected_value = apply_mutating(self.reference, operation)
                for name, peer in self.peers.items():
                    value = apply_mutating(peer, operation)
                    self.case.assertEqual(
                        value,
                        expected_value,
                        self.label(name, step, f"return value of {operation!r}"),
                    )
            self.check_full_state(step)


# -- the actual tests -----------------------------------------------------------------------------
class StateSequenceDifferentialTests(unittest.TestCase):
    def test_generated_sequences_match_the_reference_for_every_graph_type_and_seed(self) -> None:
        for directed in (False, True):
            for seed in GENERATED_SEEDS:
                with self.subTest(directed=directed, seed=seed):
                    operations = generated_sequence(seed)
                    # The sequence source must be stable: rebuilding it yields the identical list.
                    self.assertEqual(generated_sequence(seed), operations)
                    DifferentialHarness(self, directed, seed).replay(operations)

    def test_anchor_sequences_cover_every_required_shape(self) -> None:
        for tail_name, tail in ANCHOR_TAILS.items():
            for directed in (False, True):
                with self.subTest(tail=tail_name, directed=directed):
                    operations = [*ANCHOR_PREFIX, *tail]
                    DifferentialHarness(self, directed, f"anchor:{tail_name}").replay(operations)

    def test_anchor_prefix_contains_every_promised_situation(self) -> None:
        counts = Counter(operation[0] for operation in ANCHOR_PREFIX)
        self.assertGreaterEqual(counts["add_node"], 1)
        self.assertTrue(any(op[0] == "add_edge" and op[1] == op[2] for op in ANCHOR_PREFIX))
        self.assertTrue(any(op[0] == "add_edge" and op[3] == 0.0 for op in ANCHOR_PREFIX))
        pair_weights = [
            op[3] for op in ANCHOR_PREFIX if op[0] == "add_edge" and (op[1], op[2]) == ("n0", "n1")
        ]
        self.assertEqual(pair_weights, [0.0, 3, 0.0, 1])  # two overwrites, then a re-add
        self.assertIn(("remove_edge", "n3", "n4"), ANCHOR_PREFIX)
        self.assertIn(("remove_edge", "n1", "n2"), ANCHOR_PREFIX)

    def test_switch_tail_visits_consecutive_materialised_backends_and_post_switch_mutation(self) -> None:
        selections = [
            op[1]
            for op in ANCHOR_TAILS["consecutive_materialised_switches"]
            if op[0] == "configure"
        ]
        self.assertIn("dense csr dense", " ".join(selections))
        mutate_tail = ANCHOR_TAILS["switch_then_mutate"]
        self.assertTrue(any(op[0] in ("add_edge", "remove_edge", "add_node") for op in mutate_tail[1:]))


class ExceptionParityTests(unittest.TestCase):
    """The same operation must raise the same error type with the same evidence on every backend."""

    def build_family(self, directed: bool) -> dict[str, Graph]:
        rows = [("a", "b", 2.0), ("a", "a", 3.0), ("d", "e", 0.0), ("b", "c", 1.5)]
        family: dict[str, Graph] = {"adjacency": Graph(directed=directed)}
        for name in PEERS:
            family[name] = Graph(directed=directed)
        for graph in family.values():
            for source, target, weight in rows:
                graph.add_edge(source, target, weight)
            graph.add_node("iso")
        for name, peer in family.items():
            if name != "adjacency":
                peer.configure_storage(name)
        return family

    def test_invalid_weights_raise_identically_and_leave_no_trace(self) -> None:
        for directed in (False, True):
            with self.subTest(directed=directed):
                family = self.build_family(directed)
                reference = family["adjacency"]
                before = {name: read_snapshot(graph) for name, graph in family.items()}
                selected_before = {
                    name: graph.storage_document()["selected"] for name, graph in family.items()
                }
                # Algorithm answers captured once, before the rejected attempts and once after:
                # rejection must leave them (and the state they read) exactly as they were.
                shortest_before = shortest_path_results(reference)
                whole_before = whole_graph_results(reference)
                for weight in INVALID_WEIGHTS:
                    documents = {}
                    for name, graph in family.items():
                        # Both an existing pair (must not be overwritten) and a fresh pair (must
                        # add neither endpoint) are rejected.
                        with self.assertRaises(ValidationError) as existing:
                            graph.add_edge("a", "b", weight)  # type: ignore[arg-type]
                        with self.assertRaises(ValidationError) as fresh:
                            graph.add_edge("z1", "z2", weight)  # type: ignore[arg-type]
                        documents[name] = (
                            existing.exception.to_document(),
                            fresh.exception.to_document(),
                        )
                    for name in PEERS:
                        self.assertEqual(
                            documents[name],
                            documents["adjacency"],
                            f"directed={directed}, backend={name}, weight={weight!r}",
                        )
                    for name, graph in family.items():
                        self.assertEqual(
                            read_snapshot(graph),
                            before[name],
                            f"directed={directed}, backend={name}, trace left by {weight!r}",
                        )
                        self.assertEqual(graph.storage_sizes(), reference.storage_sizes())
                        # The storage selection must not move on a rejected write.
                        self.assertEqual(
                            graph.storage_document()["selected"], selected_before[name]
                        )
                self.assertEqual(shortest_path_results(reference), shortest_before)
                self.assertEqual(whole_graph_results(reference), whole_before)
                for name, peer in family.items():
                    if name == "adjacency":
                        continue
                    self.assertEqual(shortest_path_results(peer), shortest_before)
                    self.assertEqual(whole_graph_results(peer), whole_before)

    def test_empty_node_names_raise_identically_without_state_change(self) -> None:
        for directed in (False, True):
            with self.subTest(directed=directed):
                family = self.build_family(directed)
                reference = family["adjacency"]
                before = {name: read_snapshot(graph) for name, graph in family.items()}
                whole_before = whole_graph_results(reference)
                shortest_before = shortest_path_results(reference)
                documents = {"add_node": {}, "edge_source": {}, "edge_target": {}}
                for node in INVALID_NODE_NAMES:
                    for name, graph in family.items():
                        with self.assertRaises(ValidationError) as caught:
                            graph.add_node(node)
                        documents["add_node"][name] = caught.exception.to_document()
                        with self.assertRaises(ValidationError) as caught:
                            graph.add_edge(node, "a", 1.0)
                        documents["edge_source"][name] = caught.exception.to_document()
                        with self.assertRaises(ValidationError) as caught:
                            graph.add_edge("a", node, 1.0)
                        documents["edge_target"][name] = caught.exception.to_document()
                    for kind in documents:
                        for peer_name in PEERS:
                            self.assertEqual(
                                documents[kind][peer_name],
                                documents[kind]["adjacency"],
                                f"directed={directed}, backend={peer_name}, kind={kind}, node={node!r}",
                            )
                    for name, graph in family.items():
                        self.assertEqual(read_snapshot(graph), before[name])
                for name, graph in family.items():
                    self.assertEqual(whole_graph_results(graph), whole_before)
                    self.assertEqual(shortest_path_results(graph), shortest_before)

    def test_unknown_node_queries_raise_identically(self) -> None:
        for directed in (False, True):
            with self.subTest(directed=directed):
                family = self.build_family(directed)
                documents = {}
                for name, graph in family.items():
                    with self.assertRaises(ValidationError) as neighbours:
                        graph.neighbors("ghost")
                    with self.assertRaises(ValidationError) as traversal:
                        bfs(graph, "ghost")
                    with self.assertRaises(ValidationError) as ordering:
                        dfs(graph, "ghost")
                    documents[name] = (
                        neighbours.exception.to_document(),
                        traversal.exception.to_document(),
                        ordering.exception.to_document(),
                    )
                for peer_name in PEERS:
                    self.assertEqual(
                        documents[peer_name],
                        documents["adjacency"],
                        f"directed={directed}, backend={peer_name}",
                    )

    def test_negative_evidence_matches_through_repeated_switches_and_an_overwrite(self) -> None:
        for directed in (False, True):
            with self.subTest(directed=directed):
                reference = Graph(directed=directed)
                peers = {name: Graph(directed=directed) for name in PEERS}
                family = {"adjacency": reference, **peers}
                setup = [
                    ("add_edge", "a", "b", 1.0),
                    ("add_edge", "b", "c", -2.0),
                    ("add_edge", "c", "d", 1.0),
                    ("add_node", "iso"),
                ]
                switch_plan = ["csr", "dense", "auto", "adjacency", "csr"]
                step = 0
                for index, operation in enumerate(setup):
                    for graph in family.values():
                        apply_mutating(graph, operation)
                    # The negative edge (b->c) only exists from setup index 2 on.
                    if index >= 2:
                        self.assert_negative_evidence(family, step)
                    step += 1
                for requested in switch_plan:
                    reference.configure_storage("adjacency")  # the baseline never leaves adjacency
                    for name, peer in peers.items():
                        peer.configure_storage(requested)
                    self.assert_negative_evidence(family, step)
                    step += 1
                for graph in family.values():
                    apply_mutating(graph, ("add_edge", "b", "c", -2.0))  # same-shape overwrite
                self.assert_negative_evidence(family, step)

    def assert_negative_evidence(self, family: dict[str, Graph], step: int) -> None:
        def evidence(graph: Graph) -> dict:
            zero = {node: 0 for node in graph.nodes()}
            captured = {}
            for kind, runner in (
                ("dijkstra", lambda: dijkstra(graph, "a")),
                ("astar", lambda: astar(graph, "a", "d", zero)),
                ("pagerank", lambda: pagerank(graph)),
                ("communities", lambda: label_propagation(graph)),
            ):
                with self.assertRaises(NegativeWeightError) as caught:
                    runner()
                captured[kind] = caught.exception.to_document()
            return captured

        expected = evidence(family["adjacency"])
        for name, peer in family.items():
            if name == "adjacency":
                continue
            self.assertEqual(
                evidence(peer),
                expected,
                f"directed={peer.directed}, backend={name}, negative evidence diverged at step {step}",
            )


class RemoveAbsentEdgeTests(unittest.TestCase):
    def test_removing_a_non_existent_edge_keeps_returning_false_through_switches(self) -> None:
        operations = [
            ("add_edge", "a", "b", 1.0),
            ("add_node", "iso"),
            ("remove_edge", "a", "iso"),        # both nodes exist, no edge
            ("configure", "csr"),
            ("remove_edge", "x", "y"),           # neither node exists
            ("configure", "dense"),
            ("remove_edge", "b", "a"),           # the stored pair (mirror half when undirected)
            ("remove_edge", "a", "b"),           # already gone
            ("configure", "auto"),
            ("remove_edge", "iso", "a"),         # isolated node, no edge
        ]
        for directed in (False, True):
            with self.subTest(directed=directed):
                DifferentialHarness(self, directed, "anchor:remove-absent").replay(operations)
        # Pin the boolean contract at the explicitly false points on an undirected graph.
        graph = Graph()
        graph.add_edge("a", "b", 1.0)
        graph.add_node("iso")
        self.assertFalse(graph.remove_edge("a", "iso"))
        graph.configure_storage("csr")
        self.assertFalse(graph.remove_edge("x", "y"))
        self.assertTrue(graph.remove_edge("b", "a"))
        self.assertFalse(graph.remove_edge("a", "b"))

    def test_removing_a_self_loop_reports_true_once_and_then_false(self) -> None:
        for directed in (False, True):
            with self.subTest(directed=directed):
                graph = Graph(directed=directed)
                graph.add_edge("s", "s", 4.0)
                self.assertTrue(graph.remove_edge("s", "s"))
                self.assertFalse(graph.remove_edge("s", "s"))
                self.assertEqual(graph.self_loops(), [])


class ReproducibilityTests(unittest.TestCase):
    def test_generated_sequences_are_stable(self) -> None:
        first = {seed: generated_sequence(seed) for seed in GENERATED_SEEDS}
        second = {seed: generated_sequence(seed) for seed in GENERATED_SEEDS}
        self.assertEqual(second, first)
        # Distinct seeds must not all collapse onto one sequence.
        sequences = {tuple(operations) for operations in first.values()}
        self.assertGreater(len(sequences), 1)

    def test_rng_draws_are_the_pinned_lcg_sequence(self) -> None:
        rng = SequenceRng(0)
        self.assertEqual([rng.draw(10) for _ in range(6)], [3, 2, 7, 4, 7, 2])
        rng = SequenceRng(1)
        self.assertEqual([rng.draw(10) for _ in range(6)], [8, 7, 8, 5, 2, 7])

    def test_rng_draws_stay_in_range(self) -> None:
        for seed in range(20):
            rng = SequenceRng(seed)
            for bound in (1, 2, 3, 8, 16):
                for _ in range(50):
                    self.assertIn(rng.draw(bound), range(bound))


if __name__ == "__main__":
    unittest.main()
