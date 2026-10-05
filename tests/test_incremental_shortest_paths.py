"""IncrementalShortestPaths: snapshot semantics, staleness rules, diffs and backend parity.

The session snapshots Bellman-Ford at construction and must notice every later structural or
weight change -- whether it went through the session's own proxies or straight through the Graph --
while a storage-backend switch, a no-op removal and a rejected write must all leave it fresh. A
stale session refuses to answer; a successful recompute clears the staleness and returns a
deterministic diff against the last good snapshot; a negative cycle during recompute keeps the old
snapshot and the stale flag until the graph is repaired.
"""

from __future__ import annotations

import unittest

from graphtk import (
    Graph,
    IncrementalShortestPaths,
    NegativeCycleError,
    ValidationError,
    bellman_ford,
)

BACKENDS = ("adjacency", "csr", "dense", "auto")


def build(directed: bool, rows: list[tuple[str, str, float]], backend: str = "adjacency") -> Graph:
    graph = Graph(directed=directed)
    for source, target, weight in rows:
        graph.add_edge(source, target, weight)
    graph.configure_storage(backend)
    return graph


def traversable_weights(graph: Graph) -> dict[tuple[str, str], float]:
    return {
        (node, neighbour): weight
        for node in graph.nodes()
        for neighbour, weight in graph.neighbors(node)
    }


class SnapshotTests(unittest.TestCase):
    def test_initial_distances_and_paths_match_bellman_ford(self) -> None:
        rows = [("a", "b", 1), ("a", "c", 4), ("b", "c", -2), ("c", "d", 1), ("b", "d", 5)]
        graph = build(True, rows)
        graph.add_node("iso")
        distance, previous = bellman_ford(graph, "a")
        session = IncrementalShortestPaths(graph, "a")
        self.assertFalse(session.is_stale)
        self.assertEqual(session.distances, distance)
        weights = traversable_weights(graph)
        for target in distance:
            path = session.path_to(target)
            self.assertEqual(path[0], "a")
            self.assertEqual(path[-1], target)
            total = sum(weights[step] for step in zip(path, path[1:]))
            self.assertEqual(total, distance[target])

    def test_distances_returns_an_independent_copy(self) -> None:
        graph = build(False, [("a", "b", 2)])
        session = IncrementalShortestPaths(graph, "a")
        snapshot = session.distances
        snapshot["b"] = 99.0
        snapshot["ghost"] = 1.0
        self.assertEqual(session.distances, {"a": 0.0, "b": 2.0})

    def test_unreachable_and_unknown_targets(self) -> None:
        graph = build(True, [("a", "b", 3), ("c", "d", 2)])
        session = IncrementalShortestPaths(graph, "a")
        self.assertEqual(session.path_to("c"), [])  # in the graph, not reachable
        self.assertEqual(session.path_to("a"), ["a"])
        with self.assertRaises(ValidationError):
            session.path_to("ghost")

    def test_unknown_source_is_a_validation_error(self) -> None:
        graph = build(False, [("a", "b", 1)])
        with self.assertRaises(ValidationError):
            IncrementalShortestPaths(graph, "missing")

    def test_negative_cycle_at_construction_reuses_the_error_and_evidence(self) -> None:
        rows = [("a", "b", 1), ("b", "c", -2), ("c", "b", 1)]
        graph = build(True, rows)
        with self.assertRaises(NegativeCycleError) as direct:
            bellman_ford(graph, "a")
        with self.assertRaises(NegativeCycleError) as via_session:
            IncrementalShortestPaths(graph, "a")
        self.assertEqual(direct.exception.to_document(), via_session.exception.to_document())


class StalenessTests(unittest.TestCase):
    def graph(self) -> Graph:
        return build(False, [("a", "b", 1), ("b", "c", 2)])

    def assert_stale_refusal(self, session: IncrementalShortestPaths) -> None:
        with self.assertRaises(ValidationError) as from_distances:
            session.distances
        with self.assertRaises(ValidationError) as from_path:
            session.path_to("b")
        for error in (from_distances.exception, from_path.exception):
            self.assertIs(error.context["stale"], True)
            self.assertIs(error.to_document()["stale"], True)

    def test_writes_through_the_session_proxy_make_it_stale(self) -> None:
        session = IncrementalShortestPaths(self.graph(), "a")
        session.add_edge("c", "d", 4)
        self.assertTrue(session.is_stale)
        self.assert_stale_refusal(session)

    def test_writes_directly_on_the_graph_make_it_stale(self) -> None:
        graph = self.graph()
        session = IncrementalShortestPaths(graph, "a")
        graph.add_edge("a", "c", 9)
        self.assertTrue(session.is_stale)
        self.assert_stale_refusal(session)
        session.recompute()
        self.assertTrue(graph.remove_edge("a", "c"))
        self.assertTrue(session.is_stale)
        self.assert_stale_refusal(session)

    def test_a_same_weight_overwrite_still_counts_as_a_write(self) -> None:
        graph = self.graph()
        session = IncrementalShortestPaths(graph, "a")
        graph.add_edge("a", "b", 1)  # same pair, same weight: still a successful write
        self.assertTrue(session.is_stale)

    def test_removing_an_absent_edge_returns_false_and_keeps_the_session_fresh(self) -> None:
        graph = self.graph()
        session = IncrementalShortestPaths(graph, "a")
        self.assertFalse(session.remove_edge("a", "c"))
        self.assertFalse(graph.remove_edge("x", "y"))
        self.assertFalse(session.is_stale)
        self.assertEqual(session.distances, {"a": 0.0, "b": 1.0, "c": 3.0})

    def test_rejected_writes_change_neither_graph_nor_session(self) -> None:
        graph = self.graph()
        session = IncrementalShortestPaths(graph, "a")
        edges_before = [(e.source, e.target, e.weight) for e in graph.edges()]
        for bad_call in (
            lambda: session.add_edge("a", "b", float("nan")),
            lambda: session.add_edge("a", "b", "2.0"),  # type: ignore[arg-type]
            lambda: session.add_edge("", "b", 1.0),
            lambda: graph.add_edge("a", "", 1.0),
        ):
            with self.assertRaises(ValidationError):
                bad_call()
        self.assertEqual([(e.source, e.target, e.weight) for e in graph.edges()], edges_before)
        self.assertFalse(session.is_stale)
        self.assertEqual(session.distances, {"a": 0.0, "b": 1.0, "c": 3.0})

    def test_switching_the_storage_backend_is_not_a_change(self) -> None:
        graph = self.graph()
        session = IncrementalShortestPaths(graph, "a")
        for backend in ("csr", "dense", "auto", "adjacency"):
            graph.configure_storage(backend)
            self.assertFalse(session.is_stale)
            self.assertEqual(session.distances, {"a": 0.0, "b": 1.0, "c": 3.0})
            self.assertEqual(session.path_to("c"), ["a", "b", "c"])


class RecomputeTests(unittest.TestCase):
    def test_recompute_matches_a_fresh_bellman_ford_and_reports_the_diff(self) -> None:
        graph = build(True, [("a", "b", 1), ("b", "c", 2), ("x", "y", 1)])
        session = IncrementalShortestPaths(graph, "a")
        graph.add_edge("a", "d", 5)       # d becomes reachable
        graph.remove_edge("b", "c")       # c becomes unreachable
        graph.add_edge("a", "b", 7)       # b's distance changes 1 -> 7
        diff = session.recompute()
        self.assertFalse(session.is_stale)
        distance, previous = bellman_ford(graph, "a")
        self.assertEqual(session.distances, distance)
        self.assertEqual(diff, {
            "source": "a",
            "added": {"d": 5.0},
            "removed": {"c": 3.0},
            "changed": {"b": {"before": 1.0, "after": 7.0}},
        })
        for target in distance:
            self.assertEqual(session.path_to(target)[-1], target)

    def test_consecutive_changes_diff_against_the_last_successful_snapshot(self) -> None:
        graph = build(False, [("a", "b", 1)])
        session = IncrementalShortestPaths(graph, "a")
        graph.add_edge("b", "c", 2)
        graph.add_edge("c", "d", 3)  # two changes, no recompute in between
        diff = session.recompute()
        self.assertEqual(diff["added"], {"c": 3.0, "d": 6.0})
        self.assertEqual(diff["removed"], {})
        self.assertEqual(diff["changed"], {})
        # A recompute with no intervening change reports an empty diff.
        again = session.recompute()
        self.assertEqual(again, {"source": "a", "added": {}, "removed": {}, "changed": {}})

    def test_negative_cycle_keeps_the_stale_session_and_its_last_snapshot(self) -> None:
        rows = [("a", "b", 1), ("b", "c", -2), ("c", "d", 1)]
        graph = build(True, rows)
        session = IncrementalShortestPaths(graph, "a")
        self.assertEqual(session.distances, {"a": 0.0, "b": 1.0, "c": -1.0, "d": 0.0})
        graph.add_edge("d", "b", -5)  # b -> c -> d -> b now totals -6
        with self.assertRaises(NegativeCycleError) as caught:
            session.recompute()
        self.assertEqual(caught.exception.to_document()["error"], "negative_cycle_error")
        self.assertTrue(session.is_stale)
        # The old snapshot is not served as the current answer while stale.
        with self.assertRaises(ValidationError):
            session.distances
        # Repair the graph: the session recomputes against the last good snapshot.
        graph.remove_edge("d", "b")
        diff = session.recompute()
        self.assertFalse(session.is_stale)
        self.assertEqual(session.distances, {"a": 0.0, "b": 1.0, "c": -1.0, "d": 0.0})
        self.assertEqual(diff, {"source": "a", "added": {}, "removed": {}, "changed": {}})


class BackendParityTests(unittest.TestCase):
    """Diffs, path choices and error evidence must be identical on all four backends."""

    def run_scenario(self, backend: str) -> dict:
        graph = build(
            True,
            [("a", "b", 1), ("b", "c", -2), ("c", "d", 1), ("a", "d", 6), ("e", "f", 3)],
            backend,
        )
        session = IncrementalShortestPaths(graph, "a")
        record: dict[str, object] = {"initial": session.distances}
        record["paths"] = {node: session.path_to(node) for node in session.distances}
        graph.add_edge("d", "g", 2)
        graph.remove_edge("a", "b")
        with self.assertRaises(ValidationError) as stale:
            session.distances
        record["staleError"] = stale.exception.to_document()
        record["diff"] = session.recompute()
        record["final"] = session.distances
        record["finalPaths"] = {node: session.path_to(node) for node in session.distances}
        # A reachable negative cycle: the error document is part of the record too.
        graph.add_edge("g", "d", -3)
        with self.assertRaises(NegativeCycleError) as cycle:
            session.recompute()
        record["cycleError"] = cycle.exception.to_document()
        return record

    def test_all_backends_agree(self) -> None:
        reference = self.run_scenario("adjacency")
        for backend in ("csr", "dense", "auto"):
            with self.subTest(backend=backend):
                self.assertEqual(self.run_scenario(backend), reference)


if __name__ == "__main__":
    unittest.main()
