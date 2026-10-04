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
    clustering,
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

    def test_directed_components_cover_isolated_nodes(self) -> None:
        # Nodes added through add_node carry no edge; they must still each form a component.
        graph = Graph(directed=True)
        for node in ("z", "m", "q"):
            graph.add_node(node)
        self.assertEqual(components(graph), [["m"], ["q"], ["z"]])

    def test_directed_components_mix_edges_and_isolated_nodes(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("b", "a")
        graph.add_edge("c", "b")
        graph.add_node("z")
        graph.add_node("y")
        self.assertEqual(components(graph), [["a", "b", "c"], ["y"], ["z"]])
        # The union of the components is exactly the node set, each node exactly once.
        flat = [node for group in components(graph) for node in group]
        self.assertEqual(sorted(flat), graph.nodes())

    def test_directed_components_keep_nodes_orphaned_by_remove_edge(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b")
        graph.add_edge("c", "d")
        self.assertTrue(graph.remove_edge("c", "d"))
        self.assertEqual(components(graph), [["a", "b"], ["c"], ["d"]])

    def test_directed_components_self_loop_adds_no_extra_group(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b")
        graph.add_edge("s", "s")
        graph.add_node("z")
        self.assertEqual(components(graph), [["a", "b"], ["s"], ["z"]])

    def test_directed_components_do_not_depend_on_insertion_order(self) -> None:
        def build(order: list[tuple[str, object, object]]) -> Graph:
            graph = Graph(directed=True)
            for kind, left, right in order:
                if kind == "node":
                    graph.add_node(str(left))
                else:
                    graph.add_edge(str(left), str(right))
            return graph

        operations = [
            ("edge", "b", "a"),
            ("node", "z", None),
            ("edge", "c", "b"),
            ("node", "y", None),
            ("edge", "s", "s"),
        ]
        reference = components(build(operations))
        self.assertEqual(reference, [["a", "b", "c"], ["s"], ["y"], ["z"]])
        for permuted in (list(reversed(operations)), [operations[i] for i in (3, 0, 4, 1, 2)]):
            self.assertEqual(components(build(permuted)), reference)

    def test_components_leave_the_input_graph_untouched(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("b", "a", 2.5)
        graph.add_node("z")
        before = (graph.directed, graph.nodes(), [edge.to_document() for edge in graph.edges()])
        components(graph)
        after = (graph.directed, graph.nodes(), [edge.to_document() for edge in graph.edges()])
        self.assertEqual(before, after)

    def test_empty_graph_has_no_components(self) -> None:
        self.assertEqual(components(Graph()), [])
        self.assertEqual(components(Graph(directed=True)), [])

    def test_undirected_components_unchanged_by_the_fix(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b")
        graph.add_node("z")
        self.assertEqual(components(graph), [["a", "b"], ["z"]])

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


class ClusteringTests(unittest.TestCase):
    def test_a_triangle_is_fully_closed(self) -> None:
        graph = Graph()
        for source, target in (("a", "b"), ("b", "c"), ("c", "a")):
            graph.add_edge(source, target)
        result = clustering(graph)
        self.assertEqual(result.coefficients, {"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertEqual(result.average, 1.0)
        self.assertEqual(result.transitivity, 1.0)
        self.assertEqual(result.triangles, 1)
        self.assertEqual(result.connected_triples, 3)

    def test_an_open_wedge_scores_zero_but_still_counts_as_a_triple(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        result = clustering(graph)
        self.assertEqual(result.coefficients, {"a": 0.0, "b": 0.0, "c": 0.0})
        self.assertEqual(result.average, 0.0)
        self.assertEqual(result.transitivity, 0.0)
        self.assertEqual(result.triangles, 0)
        self.assertEqual(result.connected_triples, 1)

    def test_triangle_with_a_tail_mixes_closed_and_open_centres(self) -> None:
        graph = Graph()
        for source, target in (("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")):
            graph.add_edge(source, target)
        result = clustering(graph)
        self.assertEqual(result.coefficients, {"a": 1.0, "b": 1.0, "c": round(1 / 3, 10), "d": 0.0})
        self.assertEqual(result.average, round(7 / 12, 10))
        self.assertEqual(result.transitivity, 0.6)
        self.assertEqual(result.triangles, 1)
        self.assertEqual(result.connected_triples, 5)

    def test_diamond_has_two_triangles_and_eight_triples(self) -> None:
        graph = Graph()
        for source, target in (("a", "b"), ("a", "c"), ("b", "c"), ("b", "d"), ("c", "d")):
            graph.add_edge(source, target)
        result = clustering(graph)
        self.assertEqual(result.coefficients["a"], 1.0)
        self.assertEqual(result.coefficients["d"], 1.0)
        self.assertEqual(result.coefficients["b"], round(2 / 3, 10))
        self.assertEqual(result.coefficients["c"], round(2 / 3, 10))
        self.assertEqual(result.average, round(5 / 6, 10))
        self.assertEqual(result.transitivity, 0.75)
        self.assertEqual(result.triangles, 2)
        self.assertEqual(result.connected_triples, 8)

    def test_nodes_with_fewer_than_two_neighbours_and_isolated_nodes_are_zero(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b")
        graph.add_node("z")
        result = clustering(graph)
        self.assertEqual(result.coefficients, {"a": 0.0, "b": 0.0, "z": 0.0})
        self.assertEqual(result.average, 0.0)
        self.assertEqual(result.triangles, 0)
        self.assertEqual(result.connected_triples, 0)

    def test_complete_graph_of_four_counts_every_triangle_once(self) -> None:
        graph = Graph()
        nodes = ["a", "b", "c", "d"]
        for index, left in enumerate(nodes):
            for right in nodes[index + 1 :]:
                graph.add_edge(left, right)
        result = clustering(graph)
        self.assertEqual(set(result.coefficients.values()), {1.0})
        self.assertEqual(result.average, 1.0)
        self.assertEqual(result.transitivity, 1.0)
        self.assertEqual(result.triangles, 4)
        self.assertEqual(result.connected_triples, 12)

    def test_self_loops_are_neither_neighbours_nor_triangles(self) -> None:
        graph = Graph()
        graph.add_edge("a", "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("a", "c")
        result = clustering(graph)
        self.assertEqual(result.coefficients, {"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertEqual(result.triangles, 1)
        self.assertEqual(result.connected_triples, 3)

        only_loop = Graph()
        only_loop.add_edge("a", "a")
        alone = clustering(only_loop)
        self.assertEqual(alone.coefficients, {"a": 0.0})
        self.assertEqual(alone.triangles, 0)
        self.assertEqual(alone.connected_triples, 0)
        self.assertEqual(alone.average, 0.0)
        self.assertEqual(alone.transitivity, 0.0)

    def test_weights_do_not_participate(self) -> None:
        weighted = Graph()
        for source, target, weight in (("a", "b", 9.5), ("b", "c", -3.0), ("c", "a", 0.25)):
            weighted.add_edge(source, target, weight)
        plain = Graph()
        for source, target in (("a", "b"), ("b", "c"), ("c", "a")):
            plain.add_edge(source, target)
        self.assertEqual(clustering(weighted).to_document(), clustering(plain).to_document())

    def test_duplicate_undirected_input_is_one_adjacency(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("a", "b", 4.0)
        graph.add_edge("b", "a", 2.0)
        result = clustering(graph)
        self.assertEqual(result.coefficients, {"a": 0.0, "b": 0.0})
        self.assertEqual(result.connected_triples, 0)

    def test_directed_graph_uses_one_simple_undirected_reading(self) -> None:
        # One-way arcs arranged around a cycle still close one triangle in the undirected reading.
        cycle = Graph(directed=True)
        for source, target in (("a", "b"), ("b", "c"), ("c", "a")):
            cycle.add_edge(source, target)
        result = clustering(cycle)
        self.assertEqual(result.coefficients, {"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertEqual(result.triangles, 1)
        self.assertEqual(result.connected_triples, 3)

        # A single directed arc and a mutual pair must both look like exactly one undirected edge.
        single = Graph(directed=True)
        single.add_edge("a", "b")
        mutual = Graph(directed=True)
        mutual.add_edge("a", "b")
        mutual.add_edge("b", "a")
        self.assertEqual(clustering(single).to_document(), clustering(mutual).to_document())
        self.assertEqual(clustering(mutual).connected_triples, 0)

    def test_insertion_order_cannot_change_any_value(self) -> None:
        rows = [
            ("a", "b"),
            ("b", "a"),
            ("b", "c"),
            ("c", "b"),
            ("a", "c"),
            ("c", "d"),
            ("a", "a"),
        ]
        def build(order: list[int]) -> Graph:
            graph = Graph(directed=True)
            for index in order:
                graph.add_edge(*rows[index])
            return graph

        orders = [
            list(range(len(rows))),
            list(reversed(range(len(rows)))),
            [6, 0, 3, 5, 1, 4, 2],
            [2, 5, 4, 1, 6, 0, 3],
        ]
        documents = [clustering(build(order)).to_document() for order in orders]
        reference = documents[0]
        for document in documents[1:]:
            self.assertEqual(document, reference)
        self.assertEqual(reference["triangles"], 1)
        self.assertEqual(reference["connectedTriples"], 5)
        self.assertEqual(list(reference["coefficients"]), ["a", "b", "c", "d"])

    def test_isolated_nodes_are_kept_in_a_directed_graph(self) -> None:
        # A node without incident arcs must survive the undirected reading and count in the average.
        graph = Graph(directed=True)
        for source, target in (("a", "b"), ("b", "c"), ("c", "a")):
            graph.add_edge(source, target)
        graph.add_node("z")
        result = clustering(graph)
        self.assertEqual(result.coefficients, {"a": 1.0, "b": 1.0, "c": 1.0, "z": 0.0})
        self.assertEqual(result.average, 0.75)
        self.assertEqual(result.triangles, 1)
        self.assertEqual(result.connected_triples, 3)

    def test_empty_graph_is_a_complete_zero_document(self) -> None:
        result = clustering(Graph())
        self.assertEqual(
            result.to_document(),
            {
                "coefficients": {},
                "average": 0.0,
                "transitivity": 0.0,
                "triangles": 0,
                "connectedTriples": 0,
                "nodes": 0,
            },
        )

    def test_document_is_sorted_and_rounded_to_ten_decimals(self) -> None:
        graph = Graph()
        for source, target in (("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")):
            graph.add_edge(source, target)
        document = clustering(graph).to_document()
        self.assertEqual(list(document["coefficients"]), ["a", "b", "c", "d"])
        self.assertEqual(document["coefficients"]["c"], 0.3333333333)
        self.assertEqual(document["nodes"], 4)


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

    def test_staleness_survives_the_full_interleaved_sequence(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("d", "e")
        state = IncrementalComponents(graph)
        self.assertEqual(state.revision, 1)
        self.assertFalse(state.is_stale)

        # A successful removal marks its sorted endpoints stale and bumps the revision.
        self.assertTrue(state.remove_edge("b", "c"))
        self.assertEqual(state.revision, 2)
        self.assertTrue(state.is_stale)
        self.assertEqual(state.stale_nodes(), ["b", "c"])
        with self.assertRaises(ValidationError):
            state.labels()
        with self.assertRaises(ValidationError):
            state.component_count()
        document = state.to_document()
        self.assertTrue(document["stale"])
        self.assertNotIn("components", document)

        # A merging insertion (new node f joins the d-e component) completes the graph update and
        # returns the union's verdict, but must not clear or overwrite the stale markers.
        self.assertTrue(state.add_edge("e", "f"))
        self.assertEqual(state.revision, 3)
        self.assertEqual(components(state.graph), [["a", "b"], ["c"], ["d", "e", "f"]])
        self.assertTrue(state.is_stale)
        self.assertEqual(state.stale_nodes(), ["b", "c"])
        with self.assertRaises(ValidationError):
            state.labels()
        with self.assertRaises(ValidationError):
            state.component_count()
        document = state.to_document()
        self.assertTrue(document["stale"])
        self.assertNotIn("components", document)

        # Removing a missing edge returns False and changes neither revision nor the stale set.
        self.assertFalse(state.remove_edge("x", "y"))
        self.assertEqual(state.revision, 3)
        self.assertEqual(state.stale_nodes(), ["b", "c"])

        # A second successful removal while stale unions its endpoints into the stale set.
        self.assertTrue(state.remove_edge("a", "b"))
        self.assertEqual(state.revision, 4)
        self.assertEqual(state.stale_nodes(), ["a", "b", "c"])

        # recompute rebuilds from the graph as it stands, including every change made while stale.
        state.recompute()
        self.assertEqual(state.revision, 5)
        self.assertFalse(state.is_stale)
        self.assertEqual(state.stale_nodes(), [])
        groups = components(state.graph)
        self.assertEqual(groups, [["a"], ["b"], ["c"], ["d", "e", "f"]])
        expected_labels = {node: group[0] for group in groups for node in group}
        self.assertEqual(state.labels(), expected_labels)
        self.assertEqual(state.component_count(), len(groups))
        document = state.to_document()
        self.assertFalse(document["stale"])
        self.assertEqual(document["components"], 4)

    def test_a_merging_insertion_after_a_removal_cannot_clear_staleness(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("d", "e")
        state = IncrementalComponents(graph)
        self.assertTrue(state.remove_edge("b", "c"))
        # The union-find still believes b and c share a set, but c-d genuinely joins two distinct
        # union-find trees ({a,b,c} and {d,e}), so the merge reports True -- exactly the event the
        # old code treated as proof everything was current and wiped the stale markers.
        self.assertTrue(state.add_edge("c", "d"))
        self.assertTrue(state.is_stale)
        self.assertEqual(state.stale_nodes(), ["b", "c"])
        with self.assertRaises(ValidationError):
            state.component_count()
        state.recompute()
        self.assertFalse(state.is_stale)
        self.assertEqual(components(state.graph), [["a", "b"], ["c", "d", "e"]])
        self.assertEqual(state.component_count(), 2)
        self.assertEqual(set(state.labels().values()), {"a", "c"})

    def test_re_adding_the_removed_edge_does_not_heal_staleness(self) -> None:
        graph = Graph()
        for source, target in (("a", "b"), ("b", "c"), ("a", "c")):
            graph.add_edge(source, target)
        state = IncrementalComponents(graph)
        self.assertTrue(state.remove_edge("a", "b"))
        # The graph stays connected through c, so the union reports no merge -- either way, an
        # insertion is never an implicit recompute.
        self.assertFalse(state.add_edge("a", "b"))
        self.assertTrue(state.is_stale)
        self.assertEqual(state.stale_nodes(), ["a", "b"])
        state.recompute()
        self.assertFalse(state.is_stale)
        self.assertEqual(state.component_count(), 1)

    def test_revision_accounting_and_insertion_return_semantics(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b")
        state = IncrementalComponents(graph)
        self.assertEqual(state.revision, 1)
        # An intra-component edge (here a duplicate) reports no merge but is still a revision.
        self.assertFalse(state.add_edge("a", "b"))
        self.assertEqual(state.revision, 2)
        self.assertTrue(state.add_edge("b", "c"))
        self.assertEqual(state.revision, 3)
        # A failed removal is a no-op.
        self.assertFalse(state.remove_edge("a", "z"))
        self.assertEqual(state.revision, 3)
        self.assertTrue(state.remove_edge("a", "b"))
        self.assertEqual(state.revision, 4)
        # A non-merging insertion while stale still bumps the revision and leaves staleness standing.
        self.assertFalse(state.add_edge("a", "b"))
        self.assertEqual(state.revision, 5)
        self.assertTrue(state.is_stale)
        self.assertEqual(state.stale_nodes(), ["a", "b"])
        state.recompute()
        self.assertEqual(state.revision, 6)
        self.assertFalse(state.is_stale)

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
