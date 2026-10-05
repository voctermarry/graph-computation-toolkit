"""Repeatable state-sequence differential tests across the storage backends.

One graph always stays on the mutable ``adjacency`` dict and is the isomorphism reference; three
twins replay the *exact* same fixed operation sequence on ``csr``, ``dense`` and ``auto``. After
every successful step the public observation surface is compared -- nodes, edges, neighbours,
degree, edge count, self loops, the CSR export, ``to_document``, storage footprints and the
selection report -- and, whenever the state's preconditions hold, the complete results of every
algorithm (BFS, DFS, Dijkstra, Bellman-Ford, components, centrality, clustering, PageRank, label
propagation; topological sort for directed states; A* with an all-zero heuristic for reachable
targets), plus a re-walk of every ``path_from`` reconstruction. The exception type and evidence of
identical calls must also agree across backends.

The sequences come from a tiny seeded integer generator (no third-party property library, no
clock or environment input): the same seeds always pick the same cases, and a failure names the
graph type, backend, seed and the first inconsistent step. The scenarios stably cover isolated
nodes, self-loops, zero-weight edges, duplicate overwrites, removing a missing edge, edits after
a switch, same-shape weight overwrites, and repeated switches between the materialised backends.
Empty node names and non-finite/non-numeric weights keep raising ``ValidationError`` and must
change neither nodes, edges, storage selection nor subsequent algorithm results; removing a
missing edge keeps returning ``False``.
"""

from __future__ import annotations

import unittest

from graphtk import (
    CycleError,
    Graph,
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
from graphtk.graph import DENSE_LIMIT_BYTES

# Seeds are fixed lists: adding or removing a seed is an explicit, reviewable choice, never a
# random draw, so two runs of this file execute byte-for-byte identical cases. The fixed tail of
# every mixed sequence already forces each mandated scenario, so the randomised prefixes only need
# enough streams to diversify the surrounding states.
SEQUENCE_SEEDS = list(range(10))
SWITCH_SEEDS = list(range(4))
MUTATION_SEEDS = list(range(4))
BACKENDS = ("csr", "dense", "auto")
ALL_SELECTIONS = ("adjacency", "csr", "dense", "auto")
NODE_NAMES = ("a", "b", "c", "d", "e", "f")

# Finite numbers only: a generated state never approaches the 512 MiB dense ceiling (n <= 8).
WEIGHT_CHOICES = (0, 0.0, 1, 2, -2, 3, -1, 4, 0, 1, 5, -3, 2.5)

# Rejected by _validated_weight / add_node exactly the way the public API rejects them.
INVALID_WEIGHTS = (True, False, "1.5", None, float("nan"), float("inf"), float("-inf"))

# Backend-neutral keys of the storage report; requested/selected/logicalBytes are checked against
# the operation and the footprint formula instead.
NEUTRAL_STORAGE_KEYS = ("adjacencyEntries", "csrBytes", "denseBytes", "density")


class SeededChoices:
    """Deterministic integer stream -- a seeded LCG, nothing read from the clock or environment.

    Only ``randrange`` and ``choice`` are needed; the hand-rolled stream keeps case selection
    independent of hash randomisation, process seeding and the wall clock.
    """

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def randrange(self, stop: int) -> int:
        # Numerical Recipes constants; LCG low bits are weak, so consume the high half.
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 16) % stop

    def choice(self, options):
        return options[self.randrange(len(options))]


# -- operation records ----------------------------------------------------------------------------
# Every record is a tuple so a replayed step is trivially reproducible in a failure message:
#   ("node", name)                     add_node
#   ("edge", source, target, weight)   add_edge
#   ("remove", source, target)         remove_edge -> bool
#   ("storage", selection)             configure_storage -> storage document
Operation = tuple


def generate_sequence(seed: int, directed: bool) -> list[Operation]:
    """A fixed 48-step mix of edits and switches for one seed/directedness pair.

    The opening builds structure (self-loops, zero weights and duplicate pairs are biased in), the
    middle interleaves edits with backend switches, the tail stresses removals. A fixed tail then
    guarantees the required scenarios occur in every seed at the same positions.
    """
    rng = SeededChoices(seed * 2 + (1 if directed else 0) * 7919)
    operations: list[Operation] = []
    for step in range(48):
        if step < 14:
            kind = rng.choice(("node", "edge", "edge", "edge", "remove"))
        elif step < 34:
            kind = rng.choice(("node", "edge", "edge", "remove", "storage", "storage"))
        else:
            kind = rng.choice(("edge", "edge", "remove", "remove", "storage"))
        if kind == "node":
            operations.append(("node", rng.choice(NODE_NAMES)))
        elif kind == "edge":
            source = rng.choice(NODE_NAMES)
            target = rng.choice(NODE_NAMES)
            if step < 14 and rng.randrange(3) == 0:
                target = source  # reliably create self-loops in the opening
            operations.append(("edge", source, target, rng.choice(WEIGHT_CHOICES)))
        elif kind == "remove":
            operations.append(("remove", rng.choice(NODE_NAMES), rng.choice(NODE_NAMES)))
        else:
            operations.append(("storage", rng.choice(ALL_SELECTIONS)))
    # Fixed tail: the mandated scenarios appear in every sequence, at identical positions.
    operations.extend(
        [
            ("node", "iso"),  # isolated node
            ("edge", "z", "z", 0.0),  # zero-weight self-loop on a fresh node
            ("edge", "a", "b", 8.0),
            ("edge", "a", "b", 2.5),  # same shape, new weight: arrays must rebuild
            ("remove", "iso", "z"),  # never an edge: False on every backend
            ("storage", "csr"),
            ("storage", "dense"),
            ("storage", "csr"),  # consecutive switches between materialised backends
            ("edge", "d", "e", 0.0),  # edit after the switches
            ("remove", "d", "e"),
            ("storage", "auto"),
            ("node", "iso"),  # re-adding an existing node is a no-op success
        ]
    )
    return operations


def generate_switch_heavy_sequence(seed: int, directed: bool) -> list[Operation]:
    """Mostly storage switches with occasional edits: backend state survives rapid re-selection."""
    rng = SeededChoices(seed * 31 + (3 if directed else 0) * 104729)
    operations: list[Operation] = []
    for _ in range(36):
        kind = rng.choice(("storage", "storage", "storage", "edge", "remove", "node"))
        if kind == "storage":
            operations.append(("storage", rng.choice(ALL_SELECTIONS)))
        elif kind == "edge":
            operations.append(
                ("edge", rng.choice(NODE_NAMES[:4]), rng.choice(NODE_NAMES[:4]), rng.choice(WEIGHT_CHOICES))
            )
        elif kind == "remove":
            operations.append(("remove", rng.choice(NODE_NAMES[:4]), rng.choice(NODE_NAMES[:4])))
        else:
            operations.append(("node", rng.choice(NODE_NAMES[:4])))
    return operations


def generate_mutation_switch_sequence(seed: int, directed: bool) -> list[Operation]:
    """Edit on one materialised backend, switch to the other, edit, switch back: classic churn."""
    rng = SeededChoices(seed * 17 + (5 if directed else 0) * 13)
    operations: list[Operation] = [("storage", "csr")]
    for index in range(20):
        source = NODE_NAMES[rng.randrange(5)]
        target = NODE_NAMES[rng.randrange(5)]
        operations.append(("edge", source, target, rng.choice(WEIGHT_CHOICES)))
        operations.append(("storage", "dense" if index % 2 == 0 else "csr"))
        if rng.randrange(3) == 0:
            operations.append(("remove", source, target))
            operations.append(("storage", "auto"))
    return operations


# -- snapshotting ---------------------------------------------------------------------------------
def edge_tuples(graph: Graph) -> list[tuple[str, str, float]]:
    return [(edge.source, edge.target, edge.weight) for edge in graph.edges()]


def neutral_storage_report(graph: Graph) -> dict[str, object]:
    document = graph.storage_document()
    return {key: document[key] for key in NEUTRAL_STORAGE_KEYS}


def expected_selection(requested: str, n: int, csr_bytes: int, dense_bytes: int) -> str:
    """Independent re-derivation of configure_storage's resolution rule."""
    if requested != "auto":
        return requested
    selected = "csr" if n == 0 or csr_bytes <= dense_bytes else "dense"
    if selected == "dense" and dense_bytes > DENSE_LIMIT_BYTES:
        selected = "csr"
    return selected


def read_snapshot(graph: Graph) -> dict[str, object]:
    """Every public read the storage layer sits underneath (README-committed fields only)."""
    nodes = graph.nodes()
    return {
        "nodes": nodes,
        "node_count": graph.node_count,
        "edges": edge_tuples(graph),
        "edge_count": graph.edge_count(),
        "neighbors": {node: graph.neighbors(node) for node in nodes},
        "degrees": {node: graph.degree(node) for node in nodes},
        "self_loops": graph.self_loops(),
        "csr": graph.csr(),
        "to_document": graph.to_document(),
        "storage_sizes": graph.storage_sizes(),
        "storage_report": neutral_storage_report(graph),
    }


def capture(thunk):
    """Run a thunk, returning ``('ok', value)`` or ``('error', (kind, full_document))``."""
    try:
        return "ok", thunk()
    except NegativeWeightError as error:
        return "error", ("negative_weight_error", error.to_document())
    except NegativeCycleError as error:
        return "error", ("negative_cycle_error", error.to_document())
    except CycleError as error:
        return "error", ("cycle_error", error.to_document())
    except ValidationError as error:
        return "error", ("validation_error", error.to_document())


def verified_path(graph: Graph, previous, source: str, target: str, distance: float) -> list[str]:
    """Rebuild a predecessor path and prove it walks existing edges and sums to the distance."""
    path = path_from(previous, target)
    if path[0] != source or path[-1] != target:
        raise AssertionError(f"path endpoints mismatch: {path} for {source}->{target}")
    weights = {
        (node, neighbour): weight
        for node in graph.nodes()
        for neighbour, weight in graph.neighbors(node)
    }
    total = 0.0
    for step_source, step_target in zip(path, path[1:]):
        if (step_source, step_target) not in weights:
            raise AssertionError(f"path walks a missing edge: {step_source}->{step_target}")
        total += weights[(step_source, step_target)]
    if total != distance:
        raise AssertionError(f"path weight {total} != distance {distance} along {path}")
    return path


def algorithms_snapshot(graph: Graph, step: int = 0) -> dict[str, object]:
    """The algorithm surface that is defined for this state, answers and refusals alike.

    Per-source algorithms run from every node of a small graph (n <= 4: cheap even with all
    targets); on larger graphs two deterministically chosen sources suffice -- the smallest node
    as a stable anchor plus a node rotating with the step, so across a sequence every node still
    serves as a source repeatedly. Each call is captured as ``(kind, payload)``: a normal result
    or the raised error's kind and full evidence document, so "this state has no defined answer"
    and the evidence itself are compared exactly, not just successful results.
    """
    nodes = graph.nodes()
    snap: dict[str, object] = {}
    if len(nodes) <= 4:
        sources = list(nodes)
    elif nodes:
        sources = list(dict.fromkeys([nodes[0], nodes[step % len(nodes)]]))
    else:
        sources = []
    # A negative edge makes A* refuse before it ever looks at a target, so one refusal per source
    # is the complete comparison there; on a non-negative graph every reachable target is run.
    negative_graph = graph.has_negative_weights()
    for source in sources:
        results: dict[str, object] = {
            "bfs": capture(lambda s=source: bfs(graph, s)),
            "dfs": capture(lambda s=source: dfs(graph, s)),
            "dijkstra": capture(lambda s=source: dijkstra(graph, s)),
            "bellman": capture(lambda s=source: bellman_ford(graph, s)),
        }
        astar_outcomes: dict[str, object] = {}
        rebuilt_paths: dict[str, list[str]] = {}
        bellman_outcome = results["bellman"]
        if bellman_outcome[0] == "ok":
            bellman_distances, bellman_previous = bellman_outcome[1]
            if negative_graph:
                # The refusal is target-independent; capture it once against an existing node.
                astar_outcomes["__refusal__"] = capture(
                    lambda s=source: astar(graph, s, nodes[0], {node: 0.0 for node in nodes})
                )
            else:
                zero_heuristic = {node: 0.0 for node in nodes}
                for target in sorted(bellman_distances):
                    astar_outcomes[target] = capture(
                        lambda s=source, t=target, h=zero_heuristic: astar(graph, s, t, h)
                    )
                    distance, astar_previous, _ = astar_outcomes[target][1]
                    # The all-zero heuristic degenerates A* to Dijkstra: distances must agree.
                    if distance != bellman_distances[target]:
                        raise AssertionError(
                            f"astar distance {distance} != bellman distance {bellman_distances[target]}"
                        )
                    rebuilt_paths[f"astar:{target}"] = verified_path(
                        graph, astar_previous, source, target, distance
                    )
            for target in sorted(bellman_distances):
                rebuilt_paths[f"bellman:{target}"] = verified_path(
                    graph, bellman_previous, source, target, bellman_distances[target]
                )
        dijkstra_outcome = results["dijkstra"]
        if dijkstra_outcome[0] == "ok":
            dijkstra_distances, dijkstra_previous = dijkstra_outcome[1]
            for target in sorted(dijkstra_distances):
                rebuilt_paths[f"dijkstra:{target}"] = verified_path(
                    graph, dijkstra_previous, source, target, dijkstra_distances[target]
                )
        results["astar"] = astar_outcomes
        results["paths"] = rebuilt_paths
        snap[f"source:{source}"] = results
    snap["components"] = capture(lambda: components(graph))
    snap["centrality"] = capture(lambda: degree_centrality(graph))
    snap["clustering"] = capture(lambda: clustering(graph).to_document())
    snap["pagerank"] = capture(lambda: pagerank(graph).to_document())
    snap["communities"] = capture(lambda: label_propagation(graph).to_document())
    if graph.directed:
        snap["toposort"] = capture(lambda: topological_sort(graph))
    return snap


def full_snapshot(graph: Graph, step: int = 0) -> dict[str, object]:
    return {"reads": read_snapshot(graph), "algorithms": algorithms_snapshot(graph, step)}


# -- the differential harness ---------------------------------------------------------------------
class StateSequenceDifferentialTests(unittest.TestCase):
    def apply_edit(self, graph: Graph, operation: Operation):
        """Replay one edit operation, normalising a ValidationError to kind plus full evidence."""
        kind = operation[0]
        try:
            if kind == "node":
                graph.add_node(operation[1])
                return "ok", None
            if kind == "edge":
                graph.add_edge(operation[1], operation[2], operation[3])
                return "ok", None
            if kind == "remove":
                return "ok", graph.remove_edge(operation[1], operation[2])
        except ValidationError as error:
            return "error", ("validation_error", error.to_document())
        raise AssertionError(f"unknown operation kind: {kind}")

    def assert_storage_report(
        self, report: dict[str, object], graph: Graph, requested: str, context: str
    ) -> None:
        entries, csr_bytes, dense_bytes = graph.storage_sizes()
        n = graph.node_count
        selected = expected_selection(requested, n, csr_bytes, dense_bytes)
        logical_bytes = dense_bytes if selected == "dense" else csr_bytes
        self.assertEqual(report["requested"], requested, context)
        self.assertEqual(report["selected"], selected, context)
        self.assertEqual(report["logicalBytes"], logical_bytes, context)
        self.assertEqual(report["adjacencyEntries"], entries, context)
        self.assertEqual(report["csrBytes"], csr_bytes, context)
        self.assertEqual(report["denseBytes"], dense_bytes, context)
        self.assertEqual(graph._storage, selected, context)
        self.assertEqual(graph._storage_requested, requested, context)

    def run_sequence(self, operations: list[Operation], directed: bool, seed: int, label: str) -> None:
        # The reference never leaves adjacency: configure_storage steps are no-ops for it, and the
        # twins' selection reports are validated against the operation plus the footprint formula.
        reference = Graph(directed=directed)
        twins = {backend: Graph(directed=directed) for backend in BACKENDS}

        for index, operation in enumerate(operations):
            context = f"{label} directed={directed} seed={seed} step={index} op={operation!r}"
            if operation[0] == "storage":
                requested = operation[1]
                reports = {}
                for backend in BACKENDS:
                    reports[backend] = twins[backend].configure_storage(requested)
                    twin_context = f"{context} backend={backend}"
                    self.assert_storage_report(reports[backend], twins[backend], requested, twin_context)
                # The same request on identical state must yield the byte-identical report.
                self.assertEqual(reports["dense"], reports["csr"], context)
                self.assertEqual(reports["auto"], reports["csr"], context)
                # Neutral fields must match the adjacency reference, which did not switch.
                expected_report = {
                    key: reports["csr"][key] for key in NEUTRAL_STORAGE_KEYS
                }
                self.assertEqual(neutral_storage_report(reference), expected_report, context)
            else:
                reference_outcome = self.apply_edit(reference, operation)
                for backend in BACKENDS:
                    twin_context = f"{context} backend={backend}"
                    outcome = self.apply_edit(twins[backend], operation)
                    # Same success/failure shape, same evidence, same remove-edge boolean.
                    self.assertEqual(outcome, reference_outcome, twin_context)

            expected = full_snapshot(reference, index)
            for backend in BACKENDS:
                twin_context = f"{context} backend={backend}"
                snapshot = full_snapshot(twins[backend], index)
                self.assertEqual(snapshot["reads"], expected["reads"], twin_context)
                self.assertEqual(snapshot["algorithms"], expected["algorithms"], twin_context)

    # -- main scenarios ---------------------------------------------------------------------------
    def test_mixed_sequences_stay_isomorphic_on_every_backend(self) -> None:
        for directed in (False, True):
            for seed in SEQUENCE_SEEDS:
                with self.subTest(directed=directed, seed=seed):
                    self.run_sequence(generate_sequence(seed, directed), directed, seed, "mixed")

    def test_switch_heavy_sequences_stay_isomorphic(self) -> None:
        for directed in (False, True):
            for seed in SWITCH_SEEDS:
                with self.subTest(directed=directed, seed=seed):
                    self.run_sequence(
                        generate_switch_heavy_sequence(seed, directed), directed, seed, "switch-heavy"
                    )

    def test_edit_switch_churn_stays_isomorphic(self) -> None:
        for directed in (False, True):
            for seed in MUTATION_SEEDS:
                with self.subTest(directed=directed, seed=seed):
                    self.run_sequence(
                        generate_mutation_switch_sequence(seed, directed), directed, seed, "edit-switch"
                    )

    # -- pinned structural scenarios that must appear regardless of generator shape ----------------
    def test_pinned_scenarios_each_keep_all_backends_isomorphic(self) -> None:
        pinned: list[list[Operation]] = [
            [("node", "a")],  # isolated node only
            [("edge", "a", "a", 2.0)],  # self-loop (mirror side included when undirected)
            [("edge", "a", "b", 0.0), ("node", "c")],  # zero-weight edge distinct from no edge
            [("edge", "a", "b", 9.0), ("edge", "a", "b", 1.0)],  # duplicate overwrite
            # Missing edge removed, then added, really removed, removed again: F, T, F.
            [("remove", "x", "y"), ("edge", "x", "y", 1.0), ("remove", "x", "y"), ("remove", "x", "y")],
            # Same-shape weight overwrite after each materialisation.
            [("edge", "a", "b", 1.0), ("storage", "csr"), ("edge", "a", "b", 4.0)],
            [("edge", "a", "b", 1.0), ("storage", "dense"), ("edge", "a", "b", 4.0)],
            # Edits after a switch on each materialised backend.
            [("edge", "a", "b", 1.0), ("storage", "csr"), ("node", "g"), ("edge", "g", "a", 3.0), ("remove", "a", "b")],
            [("edge", "a", "b", 1.0), ("storage", "dense"), ("node", "g"), ("edge", "g", "a", 3.0), ("remove", "a", "b")],
            # Consecutive switches between the materialised backends.
            [("edge", "a", "b", 1.0), ("storage", "csr"), ("storage", "dense"), ("storage", "csr"), ("storage", "dense")],
            # auto in every transition position.
            [("edge", "a", "b", 1.0), ("storage", "auto"), ("storage", "csr"), ("storage", "auto"), ("storage", "dense")],
            # Negative edges survive re-materialisation; refusal/accept split checked afterwards.
            [("edge", "a", "b", 1.0), ("edge", "b", "c", -2.0), ("storage", "dense"), ("storage", "csr")],
            # Directed cycle (toposort refusal) and a DAG (toposort order), seen through auto.
            [("edge", "a", "b", 1.0), ("edge", "b", "c", 1.0), ("edge", "c", "a", 1.0), ("storage", "auto")],
            [("edge", "a", "b", 1.0), ("edge", "b", "c", 1.0), ("edge", "a", "c", 1.0), ("storage", "auto")],
        ]
        for directed in (False, True):
            for number, operations in enumerate(pinned):
                with self.subTest(directed=directed, scenario=number):
                    self.run_sequence(operations, directed, number, f"pinned-{number}")


class RejectionInvarianceTests(unittest.TestCase):
    """Invalid inputs raise ValidationError and leave the whole observable state untouched."""

    INVALID_EDGE_CALLS = (
        ("", "b", 1.0),
        ("a", "", 1.0),
        ("a", "b", "1.5"),
        ("a", "b", None),
        ("a", "b", True),
        ("a", "b", float("nan")),
        ("a", "b", float("inf")),
        ("a", "b", float("-inf")),
    )

    def snapshot(self, graph: Graph) -> dict[str, object]:
        return {
            "state": full_snapshot(graph),
            "storage": graph._storage,
            "requested": graph._storage_requested,
        }

    def test_rejected_calls_change_nothing_on_every_backend(self) -> None:
        for directed in (False, True):
            for backend in ALL_SELECTIONS:
                with self.subTest(directed=directed, backend=backend):
                    graph = Graph(directed=directed)
                    graph.add_edge("a", "b", 2.0)
                    graph.add_edge("b", "c", 0.0)
                    graph.add_edge("a", "a", 3.0)
                    graph.add_node("iso")
                    graph.configure_storage(backend)
                    before = self.snapshot(graph)
                    for source, target, weight in self.INVALID_EDGE_CALLS:
                        with self.assertRaises(ValidationError):
                            graph.add_edge(source, target, weight)
                    with self.assertRaises(ValidationError):
                        graph.add_node("")
                    # Nodes, edges, the active/requested storage selection and every algorithm
                    # result are exactly what they were before the rejected calls.
                    self.assertEqual(self.snapshot(graph), before)

    def test_rejection_evidence_is_identical_across_backends(self) -> None:
        for directed in (False, True):
            for source, target, weight in self.INVALID_EDGE_CALLS:
                documents = {}
                for backend in ALL_SELECTIONS:
                    graph = Graph(directed=directed)
                    graph.add_edge("a", "b", 1.0)
                    graph.configure_storage(backend)
                    with self.assertRaises(ValidationError) as caught:
                        graph.add_edge(source, target, weight)
                    documents[backend] = caught.exception.to_document()
                context = (directed, source, target, weight)
                self.assertEqual(documents["csr"], documents["adjacency"], context)
                self.assertEqual(documents["dense"], documents["adjacency"], context)
                self.assertEqual(documents["auto"], documents["adjacency"], context)

    def test_empty_node_name_is_refused_with_or_without_a_backend(self) -> None:
        # add_node never touches the read layer; the refusal and the unchanged state must be the
        # same for every selection, including after a materialised backend is active.
        for directed in (False, True):
            for backend in ALL_SELECTIONS:
                graph = Graph(directed=directed)
                graph.configure_storage(backend)
                with self.assertRaises(ValidationError):
                    graph.add_node("")
                self.assertEqual(graph.nodes(), [])

    def test_remove_missing_edge_returns_false_on_every_backend(self) -> None:
        for directed in (False, True):
            for backend in ALL_SELECTIONS:
                with self.subTest(directed=directed, backend=backend):
                    graph = Graph(directed=directed)
                    graph.add_edge("a", "b", 1.0)
                    graph.add_node("iso")
                    graph.configure_storage(backend)
                    missing_pairs = [("x", "y"), ("a", "x"), ("iso", "b")]
                    if directed:
                        missing_pairs.append(("b", "a"))  # reverse arc absent when directed
                    for source, target in missing_pairs:
                        self.assertFalse(graph.remove_edge(source, target), (backend, source, target))
                    self.assertEqual(edge_tuples(graph), [("a", "b", 1.0)])
                    self.assertEqual(graph.node_count, 3)
                    self.assertEqual(graph.self_loops(), [])

    def test_invalid_weight_types_are_all_validation_errors_and_store_nothing(self) -> None:
        for weight in INVALID_WEIGHTS:
            for directed in (False, True):
                graph = Graph(directed=directed)
                graph.configure_storage("auto")
                with self.subTest(weight=weight, directed=directed):
                    with self.assertRaises(ValidationError):
                        graph.add_edge("a", "b", weight)
                self.assertEqual(graph.nodes(), [])
                self.assertEqual(graph.edge_count(), 0)
                self.assertEqual(graph.storage_document()["selected"], "csr")  # empty auto graph


if __name__ == "__main__":
    unittest.main()
