"""Graph structure, traversal, shortest paths, centrality and incremental components."""

from __future__ import annotations

import unittest

from graphtk import (
    CycleError,
    Edge,
    Graph,
    NegativeCycleError,
    NegativeWeightError,
    ValidationError,
    bellman_ford,
    bfs,
    components,
    degree_centrality,
    dfs,
    dijkstra,
    pagerank,
    path_from,
    topological_sort,
)
from graphtk.incremental import IncrementalComponents


def sample(directed: bool = False) -> Graph:
    graph = Graph(directed=directed)
    graph.add_edge("a", "b", 1.0)
    graph.add_edge("b", "c", 2.0)
    graph.add_edge("a", "c", 5.0)
    graph.add_edge("d", "e", 1.0)
    return graph


class GraphTests(unittest.TestCase):
    def test_undirected_edges_are_symmetric(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b")
        self.assertEqual([node for node, _ in graph.neighbors("b")], ["a"])
        self.assertEqual(graph.edge_count(), 1)

    def test_directed_edges_are_one_way(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b")
        self.assertEqual([node for node, _ in graph.neighbors("b")], [])
        self.assertEqual(graph.edge_count(), 1)

    def test_last_weight_wins_and_remove_works(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("a", "b", 3.0)
        self.assertEqual(dict(graph.neighbors("a"))["b"], 3.0)
        self.assertTrue(graph.remove_edge("a", "b"))
        self.assertFalse(graph.remove_edge("a", "b"))
        self.assertEqual(graph.edge_count(), 0)

    def test_nodes_are_sorted_and_csr_offsets_agree(self) -> None:
        graph = sample()
        nodes, offsets, flat = graph.csr()
        self.assertEqual(nodes, ["a", "b", "c", "d", "e"])
        self.assertEqual(len(offsets), len(nodes) + 1)
        self.assertEqual(offsets[-1], len(flat))
        self.assertEqual(flat[offsets[0] : offsets[1]], graph.neighbors("a"))

    def test_unknown_neighbour_is_an_error(self) -> None:
        with self.assertRaises(ValidationError):
            sample().neighbors("zzz")

    def test_edge_rejects_bad_weight(self) -> None:
        with self.assertRaises(ValidationError):
            Edge("a", "b", True)  # type: ignore[arg-type]

    def test_document_reports_negative_weights_and_self_loops(self) -> None:
        graph = Graph()
        graph.add_edge("a", "a", -1.0)
        document = graph.to_document()
        self.assertEqual(document["selfLoops"], ["a"])
        self.assertTrue(document["negativeWeights"])


class TraversalTests(unittest.TestCase):
    def test_bfs_distance_and_unreachable_absent(self) -> None:
        distances = bfs(sample(), "a")
        self.assertEqual(distances, {"a": 0, "b": 1, "c": 1})

    def test_dfs_is_sorted_and_deterministic(self) -> None:
        graph = Graph(directed=True)
        for source, target in (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")):
            graph.add_edge(source, target)
        self.assertEqual(dfs(graph, "a"), ["a", "b", "d", "c"])
        self.assertEqual(dfs(graph, "a"), dfs(graph, "a"))

    def test_bfs_unknown_source(self) -> None:
        with self.assertRaises(ValidationError):
            bfs(sample(), "nope")


class ShortestPathTests(unittest.TestCase):
    def test_dijkstra_picks_the_cheaper_route(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("b", "c", 2.0)
        graph.add_edge("a", "c", 5.0)
        distance, previous = dijkstra(graph, "a")
        self.assertEqual(distance["c"], 3.0)
        self.assertEqual(path_from(previous, "c"), ["a", "b", "c"])

    def test_dijkstra_refuses_negative_weights(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", -1.0)
        with self.assertRaises(NegativeWeightError) as caught:
            dijkstra(graph, "a")
        self.assertEqual(caught.exception.context["edge"], "a->b")

    def test_bellman_ford_handles_negative_edges(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("b", "c", -2.0)
        distance, _ = bellman_ford(graph, "a")
        self.assertEqual(distance["c"], -1.0)

    def test_bellman_ford_detects_a_negative_cycle(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("b", "c", -2.0)
        graph.add_edge("c", "a", -1.0)
        with self.assertRaises(NegativeCycleError):
            bellman_ford(graph, "a")

    def test_unreachable_target_cannot_be_reconstructed(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1.0)
        graph.add_node("z")
        _, previous = dijkstra(graph, "a")
        with self.assertRaises(ValidationError):
            path_from(previous, "z")


class StructureTests(unittest.TestCase):
    def test_components_are_sorted_and_ordered(self) -> None:
        self.assertEqual(components(sample()), [["a", "b", "c"], ["d", "e"]])

    def test_components_of_a_directed_graph_use_the_weak_reading(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("b", "a")
        self.assertEqual(components(graph), [["a", "b"]])

    def test_topological_sort_respects_edges(self) -> None:
        graph = Graph(directed=True)
        for source, target in (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")):
            graph.add_edge(source, target)
        order = topological_sort(graph)
        self.assertLess(order.index("a"), order.index("d"))
        self.assertEqual(sorted(order), ["a", "b", "c", "d"])

    def test_topological_sort_reports_the_cycle(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b")
        graph.add_edge("b", "a")
        with self.assertRaises(CycleError) as caught:
            topological_sort(graph)
        self.assertEqual(caught.exception.context["cycle"], ["a", "b"])

    def test_topological_sort_needs_a_directed_graph(self) -> None:
        with self.assertRaises(ValidationError):
            topological_sort(sample())


class RankingTests(unittest.TestCase):
    def test_pagerank_converges_and_sums_to_one(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", "a")
        result = pagerank(graph)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(sum(result.scores.values()), 1.0, places=9)
        self.assertAlmostEqual(result.scores["a"], 1 / 3, places=6)

    def test_pagerank_is_deterministic(self) -> None:
        graph = sample(directed=True)
        self.assertEqual(pagerank(graph).scores, pagerank(graph).scores)

    def test_pagerank_rejects_out_of_range_damping(self) -> None:
        with self.assertRaises(ValidationError):
            pagerank(sample(), damping=1.0)

    def test_degree_centrality_is_normalised(self) -> None:
        centrality = degree_centrality(sample())
        self.assertAlmostEqual(centrality["a"], 2 / 4)
        self.assertAlmostEqual(centrality["d"], 1 / 4)


class IncrementalTests(unittest.TestCase):
    def test_insertions_merge_components_immediately(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b")
        graph.add_edge("c", "d")
        components_state = IncrementalComponents(graph)
        self.assertEqual(components_state.component_count(), 2)
        self.assertTrue(components_state.add_edge("b", "c"))
        self.assertEqual(components_state.component_count(), 1)
        self.assertFalse(components_state.add_edge("a", "d"))

    def test_removal_marks_the_labelling_stale_and_recompute_clears_it(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        state = IncrementalComponents(graph)
        state.remove_edge("a", "b")
        self.assertTrue(state.is_stale)
        self.assertEqual(state.stale_nodes(), ["a", "b"])
        with self.assertRaises(ValidationError):
            state.labels()
        state.recompute()
        self.assertFalse(state.is_stale)
        self.assertEqual(state.component_count(), 2)

    def test_directed_graphs_are_refused(self) -> None:
        with self.assertRaises(ValidationError):
            IncrementalComponents(Graph(directed=True))


class CrossCheckTests(unittest.TestCase):
    """Independent checks: they compare two answers that must agree, or verify an invariant of the
    reported answer itself. A first run of this suite was green on the weaker assertions, which is
    exactly why these exist."""

    def dense_graph(self) -> Graph:
        graph = Graph(directed=True)
        for source, target, weight in (
            ("a", "b", 2.0),
            ("a", "c", 9.0),
            ("b", "c", 1.0),
            ("b", "d", 5.0),
            ("c", "d", 2.0),
            ("d", "e", 1.0),
            ("c", "e", 7.0),
        ):
            graph.add_edge(source, target, weight)
        graph.add_node("island")
        return graph

    def test_dijkstra_and_bellman_ford_agree_on_every_reachable_node(self) -> None:
        graph = self.dense_graph()
        dj_distance, _ = dijkstra(graph, "a")
        bf_distance, _ = bellman_ford(graph, "a")
        self.assertEqual(set(dj_distance), set(bf_distance))
        for node in dj_distance:
            self.assertAlmostEqual(dj_distance[node], bf_distance[node], places=12)

    def test_reconstructed_path_weight_equals_the_reported_distance(self) -> None:
        graph = self.dense_graph()
        distance, previous = dijkstra(graph, "a")
        for target in distance:
            path = path_from(previous, target)
            self.assertEqual(path[0], "a")
            self.assertEqual(path[-1], target)
            walked = 0.0
            for step in range(len(path) - 1):
                weight = dict(graph.neighbors(path[step]))[path[step + 1]]
                walked += weight
            self.assertAlmostEqual(walked, distance[target], places=12, msg=f"path {path} does not sum to the reported distance")

    def test_unreachable_nodes_are_absent_rather_than_invented(self) -> None:
        graph = self.dense_graph()
        distance, _ = dijkstra(graph, "a")
        self.assertNotIn("island", distance)
        self.assertEqual(bfs(graph, "a").get("island"), None)

    def test_self_loop_does_not_change_distances_or_hang(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "a", 1.0)
        graph.add_edge("a", "b", 2.0)
        distance, previous = dijkstra(graph, "a")
        self.assertEqual(distance, {"a": 0.0, "b": 2.0})
        self.assertEqual(path_from(previous, "b"), ["a", "b"])

    def test_undirected_dijkstra_matches_bellman_ford(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("b", "c", 2.0)
        graph.add_edge("a", "c", 4.0)
        dj_distance, _ = dijkstra(graph, "a")
        bf_distance, _ = bellman_ford(graph, "a")
        self.assertEqual(dj_distance, bf_distance)
        self.assertEqual(dj_distance["c"], 3.0)

    def test_topological_order_is_a_valid_witness(self) -> None:
        graph = Graph(directed=True)
        for source, target in (("a", "b"), ("b", "c"), ("a", "c"), ("c", "d"), ("b", "d")):
            graph.add_edge(source, target)
        order = topological_sort(graph)
        position = {node: index for index, node in enumerate(order)}
        for edge in graph.edges():
            self.assertLess(position[edge.source], position[edge.target], msg=f"{edge.source}->{edge.target} is violated")


if __name__ == "__main__":
    unittest.main()
