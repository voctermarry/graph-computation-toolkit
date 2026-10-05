"""Tests for the IncrementalShortestPaths session.

The session is a Bellman-Ford snapshot that goes stale on every successful graph mutation and only
refreshes through ``recompute``. These tests pin: the Bellman-Ford initial semantics (negative
weights allowed, unknown source and reachable negative cycle refused with the standing evidence);
the reachable-only ``distances`` copy; ``path_to`` for reachable, present-but-unreachable and
unknown targets; change detection through both session proxies and direct graph mutation (but not
through a storage switch); stale access raising ValidationError carrying ``stale: True``; failed
writes changing nothing; the last-good-snapshot differential baseline across consecutive
mutations; recompute matching a direct Bellman-Ford run exactly; negative cycles during recompute
keeping the old snapshot and the stale state; and identical differentials, path choices and
evidence on adjacency, csr, dense and auto.
"""

from __future__ import annotations

import unittest

from graphtk import (
    Graph,
    NegativeCycleError,
    ValidationError,
    bellman_ford,
    path_from,
)
from graphtk import IncrementalShortestPaths

STORAGE_BACKENDS = ("adjacency", "csr", "dense", "auto")


def build_graph(rows: list[tuple[str, str, float]], *, directed: bool = True) -> Graph:
    graph = Graph(directed=directed)
    for source, target, weight in rows:
        graph.add_edge(source, target, weight)
    return graph


# A graph dense enough for auto to resolve to dense (9*n*n < 8*(n+1) + 16*a on a complete digraph).
def build_dense_graph() -> Graph:
    graph = Graph(directed=True)
    nodes = ["a", "b", "c", "d", "e"]
    for source in nodes:
        for target in nodes:
            if source != target:
                graph.add_edge(source, target, 1 if target > source else 3)
    return graph


def configure(graph: Graph, backend: str) -> None:
    if backend != "adjacency":
        graph.configure_storage(backend)


class ConstructionTests(unittest.TestCase):
    def test_initial_distances_and_paths_match_bellman_ford(self) -> None:
        # Non-negative rows: in an undirected reading a negative edge is a 2-edge negative cycle,
        # so directed-only battery cases cover negative weights.
        rows = [("a", "b", 1), ("a", "c", 4), ("b", "c", 2), ("c", "d", 1), ("e", "f", 3)]
        for directed in (False, True):
            with self.subTest(directed=directed):
                graph = build_graph(rows, directed=directed)
                expected_distance, expected_previous = bellman_ford(graph, "a")
                session = IncrementalShortestPaths(graph, "a")
                self.assertFalse(session.is_stale)
                self.assertEqual(session.distances(), expected_distance)
                for target in expected_distance:
                    self.assertEqual(
                        session.path_to(target),
                        path_from(expected_previous, target),
                    )

    def test_finite_negative_weights_are_allowed(self) -> None:
        graph = build_graph([("a", "b", 1), ("b", "c", -5), ("c", "d", 2)])
        session = IncrementalShortestPaths(graph, "a")
        self.assertEqual(session.distances(), {"a": 0.0, "b": 1.0, "c": -4.0, "d": -2.0})

    def test_unknown_source_is_a_validation_error(self) -> None:
        graph = build_graph([("a", "b", 1)])
        with self.assertRaises(ValidationError) as caught:
            IncrementalShortestPaths(graph, "missing")
        # Same evidence the standalone source check produces.
        self.assertEqual(caught.exception.to_document(), bellman_ford_unknown_source_document(graph))

    def test_reachable_negative_cycle_at_construction_raises_standing_evidence(self) -> None:
        rows = [("a", "b", 1), ("b", "c", -2), ("c", "b", 1)]
        for backend in STORAGE_BACKENDS:
            graph = build_graph(rows)
            configure(graph, backend)
            with self.assertRaises(NegativeCycleError) as direct:
                bellman_ford(graph, "a")
            with self.assertRaises(NegativeCycleError) as session:
                IncrementalShortestPaths(graph, "a")
            self.assertEqual(
                session.exception.to_document(),
                direct.exception.to_document(),
                backend,
            )

    def test_unreachable_negative_cycle_does_not_block_construction(self) -> None:
        # b<->c is a negative 2-edge cycle but cannot be reached from a.
        graph = build_graph([("a", "a", 1), ("b", "c", 1), ("c", "b", -3)])
        session = IncrementalShortestPaths(graph, "a")
        self.assertEqual(session.distances(), {"a": 0.0})
        self.assertEqual(session.path_to("b"), [])


def bellman_ford_unknown_source_document(graph: Graph) -> dict:
    try:
        bellman_ford(graph, "missing")
    except ValidationError as error:
        return error.to_document()
    raise AssertionError("bellman_ford accepted an unknown source")


class DistancesAndPathsTests(unittest.TestCase):
    def setUp(self) -> None:
        # a reaches b, c, d; e-f is a separate component.
        self.graph = build_graph(
            [("a", "b", 1), ("a", "c", 4), ("b", "c", 1), ("c", "d", 2), ("e", "f", 3)]
        )
        self.session = IncrementalShortestPaths(self.graph, "a")

    def test_distances_contain_only_reachable_nodes(self) -> None:
        self.assertEqual(self.session.distances(), {"a": 0.0, "b": 1.0, "c": 2.0, "d": 4.0})

    def test_distances_returns_a_copy(self) -> None:
        distances = self.session.distances()
        distances["a"] = 999.0
        distances["zz"] = 0.0
        self.assertEqual(self.session.distances(), {"a": 0.0, "b": 1.0, "c": 2.0, "d": 4.0})

    def test_path_to_source_is_just_the_source(self) -> None:
        self.assertEqual(self.session.path_to("a"), ["a"])

    def test_path_to_reachable_target_walks_real_edges_summing_to_the_distance(self) -> None:
        weights = {
            (node, neighbour): weight
            for node in self.graph.nodes()
            for neighbour, weight in self.graph.neighbors(node)
        }
        for target, distance in self.session.distances().items():
            path = self.session.path_to(target)
            self.assertEqual(path[0], "a")
            self.assertEqual(path[-1], target)
            total = 0.0
            for left, right in zip(path, path[1:]):
                self.assertIn((left, right), weights)
                total += weights[(left, right)]
            self.assertEqual(total, distance)

    def test_path_to_present_but_unreachable_node_is_empty(self) -> None:
        self.assertEqual(self.session.path_to("e"), [])
        self.assertEqual(self.session.path_to("f"), [])

    def test_path_to_unknown_node_is_a_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            self.session.path_to("ghost")


class ChangeDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph([("a", "b", 1), ("a", "c", 4)])

    def test_session_add_edge_marks_the_snapshot_stale(self) -> None:
        session = IncrementalShortestPaths(self.graph, "a")
        session.add_edge("b", "c", -2)
        self.assertTrue(session.is_stale)
        self.assert_stale_error(session.distances)
        self.assert_stale_error(lambda: session.path_to("c"))

    def test_session_remove_edge_marks_the_snapshot_stale(self) -> None:
        session = IncrementalShortestPaths(self.graph, "a")
        self.assertTrue(session.remove_edge("a", "b"))
        self.assertTrue(session.is_stale)
        self.assert_stale_error(session.distances)
        self.assert_stale_error(lambda: session.path_to("b"))

    def test_direct_graph_mutations_are_detected(self) -> None:
        session = IncrementalShortestPaths(self.graph, "a")
        self.graph.add_edge("b", "c", 1)
        self.assertTrue(session.is_stale)
        session.recompute()
        self.graph.remove_edge("a", "b")
        self.assertTrue(session.is_stale)
        session.recompute()
        self.graph.add_node("isolated")
        self.assertTrue(session.is_stale)

    def test_storage_switch_is_not_a_change(self) -> None:
        session = IncrementalShortestPaths(self.graph, "a")
        before = session.distances()
        for backend in ("csr", "dense", "auto", "adjacency"):
            self.graph.configure_storage(backend)
            self.assertFalse(session.is_stale, backend)
            self.assertEqual(session.distances(), before, backend)
            self.assertEqual(session.path_to("c"), ["a", "c"], backend)

    def test_storage_switch_between_mutations_does_not_heal_staleness(self) -> None:
        session = IncrementalShortestPaths(self.graph, "a")
        session.add_edge("b", "d", 2)
        for backend in ("csr", "dense", "auto", "adjacency"):
            self.graph.configure_storage(backend)
            self.assertTrue(session.is_stale, backend)
            self.assert_stale_error(session.distances)

    def test_removing_a_non_existent_edge_returns_false_and_changes_nothing(self) -> None:
        session = IncrementalShortestPaths(self.graph, "a")
        distances = session.distances()
        self.assertFalse(session.remove_edge("a", "missing"))
        self.assertFalse(session.remove_edge("x", "y"))
        self.assertFalse(session.is_stale)
        self.assertEqual(session.distances(), distances)

    def test_rejected_weight_changes_neither_graph_nor_session(self) -> None:
        session = IncrementalShortestPaths(self.graph, "a")
        nodes, edges = self.graph.nodes(), self.graph.edges()
        for bad in (float("nan"), float("inf"), "2.0", None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    session.add_edge("a", "b", bad)  # type: ignore[arg-type]
                self.assertFalse(session.is_stale)
                self.assertEqual(self.graph.nodes(), nodes)
                self.assertEqual(self.graph.edges(), edges)
                self.assertEqual(session.distances(), {"a": 0.0, "b": 1.0, "c": 4.0})
        # A fresh pair with a bad weight leaves neither endpoint behind.
        with self.assertRaises(ValidationError):
            session.add_edge("z1", "z2", "nope")  # type: ignore[arg-type]
        self.assertEqual(self.graph.nodes(), nodes)

    def test_rejected_node_name_changes_neither_graph_nor_session(self) -> None:
        session = IncrementalShortestPaths(self.graph, "a")
        nodes, edges = self.graph.nodes(), self.graph.edges()
        with self.assertRaises(ValidationError):
            session.add_edge("", "b", 1.0)
        with self.assertRaises(ValidationError):
            session.add_edge("a", "", 1.0)
        self.assertFalse(session.is_stale)
        self.assertEqual(self.graph.nodes(), nodes)
        self.assertEqual(self.graph.edges(), edges)

    def assert_stale_error(self, call) -> None:
        with self.assertRaises(ValidationError) as caught:
            call()
        self.assertIs(caught.exception.context["stale"], True)
        self.assertIs(caught.exception.to_document()["stale"], True)


class RecomputeTests(unittest.TestCase):
    def test_recompute_matches_direct_bellman_ford_and_clears_stale(self) -> None:
        graph = build_graph([("a", "b", 1), ("a", "c", 4)])
        session = IncrementalShortestPaths(graph, "a")
        session.add_edge("b", "c", -2)
        session.add_edge("c", "d", 1)
        document = session.recompute()
        self.assertFalse(session.is_stale)
        distance, previous = bellman_ford(graph, "a")
        self.assertEqual(session.distances(), distance)
        for target in distance:
            self.assertEqual(session.path_to(target), path_from(previous, target))
        # Only c changed (4 -> -1) and d is new; b and a stay out of the document.
        self.assertEqual(
            document,
            {
                "source": "a",
                "added": {"d": 0.0},
                "removed": {},
                "changed": {"c": {"before": 4.0, "after": -1.0}},
            },
        )

    def test_added_removed_and_changed_sections(self) -> None:
        graph = build_graph([("a", "b", 1), ("b", "c", 2)])
        session = IncrementalShortestPaths(graph, "a")
        self.assertEqual(session.distances(), {"a": 0.0, "b": 1.0, "c": 3.0})

        # Remove the only route to b/c: both disappear; a new edge a->d 5 adds d.
        graph.remove_edge("a", "b")
        graph.add_edge("a", "d", 5)
        document = session.recompute()
        self.assertEqual(document["source"], "a")
        self.assertEqual(document["added"], {"d": 5.0})
        self.assertEqual(document["removed"], {"b": 1.0, "c": 3.0})
        self.assertEqual(document["changed"], {})
        self.assertEqual(session.distances(), {"a": 0.0, "d": 5.0})

    def test_consecutive_mutations_diff_against_the_last_good_snapshot(self) -> None:
        graph = build_graph([("a", "b", 1), ("b", "c", 2)])
        session = IncrementalShortestPaths(graph, "a")  # b=1, c=3

        # Two mutations with no recompute between them; neither old answer is ever served.
        graph.remove_edge("a", "b")
        graph.add_edge("a", "c", 2)
        self.assertTrue(session.is_stale)
        document = session.recompute()
        # b became unreachable (removed), c shortened 3 -> 2 (changed): one combined differential.
        self.assertEqual(document["added"], {})
        self.assertEqual(document["removed"], {"b": 1.0})
        self.assertEqual(document["changed"], {"c": {"before": 3.0, "after": 2.0}})
        self.assertEqual(session.distances(), {"a": 0.0, "c": 2.0})

        # A recompute with nothing changed reports an empty differential.
        self.assertEqual(
            session.recompute(),
            {"source": "a", "added": {}, "removed": {}, "changed": {}},
        )

    def test_node_order_in_the_document_is_stable(self) -> None:
        graph = build_graph([("a", "b", 1)])
        session = IncrementalShortestPaths(graph, "a")
        for target, weight in (("z", 1), ("m", 1), ("d", 1), ("b", 5)):
            graph.add_edge("a", target, weight)
        document = session.recompute()
        # Added nodes appear in node-name order regardless of insertion order; b (existing) changed.
        self.assertEqual(list(document["added"]), ["d", "m", "z"])
        self.assertEqual(list(document["changed"]), ["b"])

        graph.remove_edge("a", "z")
        graph.remove_edge("a", "m")
        graph.remove_edge("a", "d")
        document = session.recompute()
        self.assertEqual(list(document["removed"]), ["d", "m", "z"])

    def test_negative_cycle_on_recompute_keeps_stale_and_the_last_snapshot(self) -> None:
        graph = build_graph([("a", "b", 1), ("b", "c", 2)])
        session = IncrementalShortestPaths(graph, "a")
        good_distances = {"a": 0.0, "b": 1.0, "c": 3.0}
        self.assertEqual(session.distances(), good_distances)

        # Introduce a reachable negative cycle: b->c 2 plus c->b -4 totals -2.
        graph.add_edge("c", "b", -4)
        with self.assertRaises(NegativeCycleError) as direct:
            bellman_ford(graph, "a")
        with self.assertRaises(NegativeCycleError) as caught:
            session.recompute()
        self.assertEqual(caught.exception.to_document(), direct.exception.to_document())
        self.assertTrue(session.is_stale)
        with self.assertRaises(ValidationError) as stale_distances:
            session.distances()
        self.assertIs(stale_distances.exception.to_document()["stale"], True)
        with self.assertRaises(ValidationError):
            session.path_to("c")

        # Repair the graph, then recompute succeeds -- and diffs against the pre-cycle snapshot,
        # which proves that snapshot was never overwritten (b=1, c=3 unchanged -> empty sections).
        graph.remove_edge("c", "b")
        document = session.recompute()
        self.assertFalse(session.is_stale)
        self.assertEqual(session.distances(), good_distances)
        self.assertEqual(document, {"source": "a", "added": {}, "removed": {}, "changed": {}})

    def test_failed_recompute_can_be_retried_after_each_further_repair(self) -> None:
        graph = build_graph([("a", "b", 1)])
        session = IncrementalShortestPaths(graph, "a")
        graph.add_edge("b", "c", -2)
        graph.add_edge("c", "b", 1)
        for _ in range(2):
            with self.assertRaises(NegativeCycleError):
                session.recompute()
            self.assertTrue(session.is_stale)
        graph.remove_edge("c", "b")
        session.recompute()
        self.assertFalse(session.is_stale)
        self.assertEqual(session.distances(), {"a": 0.0, "b": 1.0, "c": -1.0})


def assert_stale(session: IncrementalShortestPaths) -> None:
    with pytest_raises_stale(session.distances):
        pass


class _StaleMarker:
    pass


def pytest_raises_stale(call):
    import contextlib

    return contextlib.contextmanager(lambda: (yield))()  # pragma: no cover - replaced below


class StorageConsistencyTests(unittest.TestCase):
    def reference_documents(self, rows: list[tuple], operations: list[tuple]) -> list[dict]:
        """The adjacency-backed differential at every step, the canonical answer for peer backends."""
        graph = build_graph(rows)
        session = IncrementalShortestPaths(graph, "a")
        documents: list[dict] = []
        for operation in operations:
            kind = operation[0]
            if kind == "add_edge":
                graph.add_edge(operation[1], operation[2], operation[3])
            elif kind == "remove_edge":
                graph.remove_edge(operation[1], operation[2])
            elif kind == "add_node":
                graph.add_node(operation[1])
            documents.append(session.recompute())
        return documents

    def test_documents_and_paths_are_identical_on_every_backend(self) -> None:
        rows = [("a", "b", 2), ("a", "c", 5), ("b", "c", -1), ("c", "d", 3), ("e", "f", 1)]
        operations = [
            ("add_edge", "b", "d", 4),
            ("remove_edge", "a", "c"),
            ("add_edge", "d", "c", -2),
            ("add_edge", "a", "g", 7),
            ("add_node", "iso"),
            ("remove_edge", "e", "f"),
            ("add_edge", "g", "d", -3),
            ("remove_edge", "b", "d"),
        ]
        reference_documents = self.reference_documents(rows, operations)
        for backend in STORAGE_BACKENDS:
            with self.subTest(backend=backend):
                graph = build_graph(rows)
                configure(graph, backend)
                session = IncrementalShortestPaths(graph, "a")
                for step, operation in enumerate(operations):
                    kind = operation[0]
                    if kind == "add_edge":
                        session.add_edge(operation[1], operation[2], operation[3])
                    elif kind == "remove_edge":
                        session.remove_edge(operation[1], operation[2])
                    elif kind == "add_node":
                        graph.add_node(operation[1])
                    document = session.recompute()
                    self.assertEqual(document, reference_documents[step], f"step {step}")
                    self.assertFalse(session.is_stale)
                    distance, previous = bellman_ford(graph, "a")
                    self.assertEqual(session.distances(), distance)
                    for target in distance:
                        self.assertEqual(
                            session.path_to(target),
                            path_from(previous, target),
                            f"step {step}, target {target}",
                        )
                    # Present but unreachable nodes still return the empty list on every backend.
                    for node in graph.nodes():
                        if node not in distance:
                            self.assertEqual(session.path_to(node), [])

    def test_auto_resolves_to_dense_and_still_agrees(self) -> None:
        for backend in STORAGE_BACKENDS:
            with self.subTest(backend=backend):
                graph = build_dense_graph()
                configure(graph, backend)
                if backend == "auto":
                    self.assertEqual(graph.storage_document()["selected"], "dense")
                session = IncrementalShortestPaths(graph, "a")
                distance, previous = bellman_ford(graph, "a")
                self.assertEqual(session.distances(), distance)
                for target in distance:
                    self.assertEqual(session.path_to(target), path_from(previous, target))
                # A negative-weight write after the snapshot: recompute must match adjacency exactly.
                graph.add_edge("a", "e", -1)
                document = session.recompute()
                reference = build_dense_graph()
                reference.add_edge("a", "e", -1)
                self.assertEqual(session.distances(), bellman_ford(reference, "a")[0])
                self.assertIn("e", document["changed"])

    def test_negative_cycle_evidence_is_identical_on_every_backend(self) -> None:
        rows = [("a", "b", 1), ("b", "c", 2)]
        documents = {}
        for backend in STORAGE_BACKENDS:
            graph = build_graph(rows)
            configure(graph, backend)
            session = IncrementalShortestPaths(graph, "a")
            graph.add_edge("c", "b", -4)
            with self.assertRaises(NegativeCycleError) as caught:
                session.recompute()
            documents[backend] = caught.exception.to_document()
            self.assertTrue(session.is_stale)
        for backend in ("csr", "dense", "auto"):
            self.assertEqual(documents[backend], documents["adjacency"])

    def test_undirected_graphs_are_supported_with_the_same_semantics(self) -> None:
        # No negative edges: in an undirected graph each such edge closes a reachable negative cycle.
        rows = [("a", "b", 2), ("b", "c", 3), ("a", "c", 5), ("c", "d", 1)]
        for backend in STORAGE_BACKENDS:
            with self.subTest(backend=backend):
                graph = build_graph(rows, directed=False)
                configure(graph, backend)
                session = IncrementalShortestPaths(graph, "a")
                distance, previous = bellman_ford(graph, "a")
                self.assertEqual(session.distances(), distance)
                for target in distance:
                    self.assertEqual(session.path_to(target), path_from(previous, target))


if __name__ == "__main__":
    unittest.main()
