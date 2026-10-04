"""Regression tests for the single weight-validation rule shared by every construction entry.

Every public way to put a weight into a graph -- ``Edge``, ``Graph.add_edge`` and
``IncrementalComponents.add_edge`` -- must accept only finite ``int``/``float`` values, reject
``bool``/``str``/``None``/non-finite floats with ``ValidationError``, and leave no state behind
when it rejects. Legal negative weights remain legal for Bellman-Ford.
"""

from __future__ import annotations

import math
import sys
import unittest

from graphtk import Edge, Graph, NegativeWeightError, ValidationError, bellman_ford, dijkstra
from graphtk.incremental import IncrementalComponents

INVALID_WEIGHTS = [
    True,
    False,
    "1.5",  # float("1.5") used to be accepted by Graph.add_edge
    "nan",
    "inf",
    "",
    None,
    float("nan"),
    float("inf"),
    float("-inf"),
]

LEGAL_WEIGHTS = [
    0,
    0.0,
    -0.0,
    3,
    -7,
    0.25,
    -2.5,
    1e308,  # largest finite float
    sys.float_info.min,
    sys.float_info.min / 2,  # a positive subnormal: finite, tiny but real
]


class EdgeValidationTests(unittest.TestCase):
    def test_invalid_weights_are_rejected(self) -> None:
        for weight in INVALID_WEIGHTS:
            with self.subTest(weight=weight):
                with self.assertRaises(ValidationError):
                    Edge("a", "b", weight)  # type: ignore[arg-type]

    def test_legal_weights_are_accepted_and_normalised_to_float(self) -> None:
        for weight in LEGAL_WEIGHTS:
            with self.subTest(weight=weight):
                edge = Edge("a", "b", weight)
                self.assertIsInstance(edge.weight, float)
                self.assertEqual(edge.weight, float(weight))
                self.assertTrue(math.isfinite(edge.weight))

    def test_negative_weight_stays_legal(self) -> None:
        self.assertEqual(Edge("a", "b", -3).weight, -3.0)

    def test_default_weight_is_one(self) -> None:
        self.assertEqual(Edge("a", "b").weight, 1.0)

    def test_rejected_edge_is_not_constructed(self) -> None:
        with self.assertRaises(ValidationError):
            Edge("", "b", 1.0)
        with self.assertRaises(ValidationError):
            Edge("a", "b", float("nan"))


class AddEdgeValidationTests(unittest.TestCase):
    def test_invalid_weights_are_rejected(self) -> None:
        for directed in (False, True):
            for weight in INVALID_WEIGHTS:
                with self.subTest(directed=directed, weight=weight):
                    graph = Graph(directed=directed)
                    with self.assertRaises(ValidationError):
                        graph.add_edge("a", "b", weight)  # type: ignore[arg-type]

    def test_legal_weights_are_accepted_and_normalised_to_float(self) -> None:
        for weight in LEGAL_WEIGHTS:
            graph = Graph()
            graph.add_edge("a", "b", weight)
            stored = dict(graph.neighbors("a"))["b"]
            self.assertIsInstance(stored, float)
            self.assertEqual(stored, float(weight))

    def test_rejected_weight_adds_no_endpoints(self) -> None:
        graph = Graph()
        with self.assertRaises(ValidationError):
            graph.add_edge("a", "b", float("nan"))
        self.assertEqual(graph.nodes(), [])
        self.assertEqual(graph.node_count, 0)
        self.assertEqual(graph.edges(), [])

    def test_rejected_weight_does_not_overwrite_an_existing_edge(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b", 2.0)
        with self.assertRaises(ValidationError):
            graph.add_edge("a", "b", "inf")  # type: ignore[arg-type]
        self.assertEqual(dict(graph.neighbors("a"))["b"], 2.0)
        self.assertEqual(dict(graph.neighbors("b"))["a"], 2.0)
        self.assertEqual(graph.edge_count(), 1)

    def test_undirected_rejection_leaves_no_half_edge_and_no_endpoints(self) -> None:
        graph = Graph()
        with self.assertRaises(ValidationError):
            graph.add_edge("a", "b", "1.5")  # type: ignore[arg-type]
        # Neither endpoint may exist, and adjacency must not appear on either side.
        self.assertEqual(graph.nodes(), [])
        self.assertNotIn("a", graph._adjacency)
        self.assertNotIn("b", graph._adjacency)

    def test_undirected_rejection_beside_an_existing_node_leaves_no_half_edge(self) -> None:
        graph = Graph()
        graph.add_edge("a", "z", 1.0)
        with self.assertRaises(ValidationError):
            graph.add_edge("a", "b", True)  # type: ignore[arg-type]
        # Existing state is intact, the new endpoint was never created.
        self.assertEqual(graph.nodes(), ["a", "z"])
        self.assertEqual(dict(graph.neighbors("a")), {"z": 1.0})
        self.assertNotIn("b", graph._adjacency)

    def test_undirected_sides_share_the_same_weight(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b", -4)
        self.assertEqual(dict(graph.neighbors("a"))["b"], -4.0)
        self.assertEqual(dict(graph.neighbors("b"))["a"], -4.0)

    def test_duplicate_legal_edge_still_updates_weight(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("a", "b", 9.0)
        self.assertEqual(dict(graph.neighbors("a"))["b"], 9.0)
        self.assertEqual(graph.edge_count(), 1)

    def test_integer_weights_work_throughout_the_algorithms(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1)
        graph.add_edge("b", "c", 2)
        distance, _ = dijkstra(graph, "a")
        self.assertEqual(distance["c"], 3.0)
        self.assertIsInstance(distance["c"], float)

    def test_negative_weights_reach_bellman_ford_but_not_dijkstra(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1)
        graph.add_edge("b", "c", -2)
        distance, _ = bellman_ford(graph, "a")
        self.assertEqual(distance["c"], -1.0)
        with self.assertRaises(NegativeWeightError):
            dijkstra(graph, "a")


class IncrementalAtomicityTests(unittest.TestCase):
    def _state(self) -> tuple[Graph, IncrementalComponents]:
        graph = Graph()
        graph.add_edge("a", "b")
        graph.add_edge("c", "d")
        return graph, IncrementalComponents(graph)

    def test_rejected_insertion_changes_nothing(self) -> None:
        graph, state = self._state()
        revision = state.revision
        labels = state.labels()
        stale = state.stale_nodes()
        nodes, edges = graph.nodes(), graph.edges()
        for weight in INVALID_WEIGHTS:
            with self.subTest(weight=weight):
                with self.assertRaises(ValidationError):
                    state.add_edge("x", "y", weight)  # type: ignore[arg-type]
                self.assertEqual(state.revision, revision)
                self.assertEqual(state.stale_nodes(), stale)
                self.assertEqual(state.labels(), labels)
                self.assertEqual(graph.nodes(), nodes)
                self.assertEqual(graph.edges(), edges)
                self.assertEqual(state.component_count(), 2)

    def test_rejected_insertion_between_existing_nodes_changes_nothing(self) -> None:
        graph, state = self._state()
        revision = state.revision
        # Endpoints that already exist: the failure must not overwrite the (absent) edge or bump
        # revision, and the union-find must not have merged the two components.
        with self.assertRaises(ValidationError):
            state.add_edge("b", "c", float("inf"))
        self.assertEqual(state.revision, revision)
        self.assertFalse(state.is_stale)
        self.assertEqual(state.component_count(), 2)
        self.assertEqual(graph.edge_count(), 2)
        self.assertEqual(dict(graph.neighbors("b")), {"a": 1.0})

    def test_rejected_insertion_while_stale_leaves_the_stale_set_alone(self) -> None:
        graph, state = self._state()
        self.assertTrue(state.remove_edge("a", "b"))
        revision = state.revision
        with self.assertRaises(ValidationError):
            state.add_edge("b", "c", "nan-string")  # type: ignore[arg-type]
        self.assertEqual(state.revision, revision)
        self.assertTrue(state.is_stale)
        self.assertEqual(state.stale_nodes(), ["a", "b"])
        self.assertEqual(graph.nodes(), ["a", "b", "c", "d"])
        self.assertEqual(graph.edge_count(), 1)

    def test_successful_insertion_still_unions_and_bumps_revision(self) -> None:
        graph, state = self._state()
        self.assertTrue(state.add_edge("b", "c", -5))
        self.assertEqual(state.revision, 2)
        self.assertEqual(state.component_count(), 1)
        self.assertEqual(dict(graph.neighbors("b"))["c"], -5.0)
        self.assertEqual(dict(graph.neighbors("c"))["b"], -5.0)


if __name__ == "__main__":
    unittest.main()
