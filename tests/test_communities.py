"""Deterministic weighted label propagation: the API contract and its byte-stable CLI output.

The update rule -- weighted majority with "keep the current label on a tie it belongs to" -- is
exercised on hand-traced graphs, and the determinism promise is checked the way the user hits it:
the same graph fed through the ``communities`` command with its edge rows shuffled must produce
byte-identical stdout.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock

from graphtk import CommunityResult, Graph, NegativeWeightError, ValidationError, label_propagation
from graphtk.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main


def edges(*rows: tuple[str, str, float]) -> Graph:
    graph = Graph()
    for source, target, weight in rows:
        graph.add_edge(source, target, weight)
    return graph


class LabelPropagationAPITests(unittest.TestCase):
    def test_empty_graph_is_the_full_zero_document(self) -> None:
        result = label_propagation(Graph())
        self.assertIsInstance(result, CommunityResult)
        self.assertEqual(result.communities, [])
        self.assertEqual(result.labels, {})
        self.assertEqual(result.count, 0)
        self.assertEqual(result.iterations, 0)
        self.assertTrue(result.converged)
        self.assertEqual(
            result.to_document(),
            {"communities": [], "count": 0, "converged": True, "iterations": 0, "labels": {}},
        )

    def test_single_node_is_one_singleton_community_converged_in_one_round(self) -> None:
        graph = Graph()
        graph.add_node("a")
        result = label_propagation(graph)
        self.assertEqual(result.communities, [["a"]])
        self.assertEqual(result.labels, {"a": "a"})
        self.assertEqual((result.count, result.iterations, result.converged), (1, 1, True))

    def test_isolated_nodes_keep_singleton_communities(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 2)
        graph.add_node("z")  # added on its own
        graph.add_edge("c", "c", 5)  # self-loop only: ignored, so c is isolated too
        result = label_propagation(graph)
        self.assertEqual(result.communities, [["a", "b"], ["c"], ["z"]])
        self.assertEqual(result.labels, {"a": "a", "b": "a", "c": "c", "z": "z"})
        self.assertEqual(result.count, 3)
        self.assertTrue(result.converged)

    def test_two_weight_dense_clusters_separate(self) -> None:
        graph = edges(
            ("a", "b", 3), ("a", "c", 2), ("b", "c", 2),
            ("c", "d", 1),
            ("d", "e", 2), ("d", "f", 2), ("e", "f", 3),
        )
        result = label_propagation(graph)
        self.assertEqual(result.communities, [["a", "b", "c"], ["d", "e", "f"]])
        self.assertEqual(result.labels, {"a": "a", "b": "a", "c": "a", "d": "d", "e": "d", "f": "d"})
        self.assertTrue(result.converged)

    def test_tie_keeps_the_current_label_when_it_is_among_the_best(self) -> None:
        # Edges: a-c=4, a-d=1, b-d=1. Round 1: a adopts c (4 beats 1), b adopts d, c sees a already
        # labelled c and stays. d then scores label c (through a) and label d (through b) at 1 each;
        # its own label d is one of the tied labels, so it keeps d. Round 2 changes nothing. Without
        # retention d would take the lex-min label c and the whole path would collapse to one group.
        graph = edges(("a", "c", 4), ("a", "d", 1), ("b", "d", 1))
        result = label_propagation(graph)
        self.assertEqual(result.communities, [["a", "c"], ["b", "d"]])
        self.assertEqual(result.labels, {"a": "a", "b": "b", "c": "a", "d": "b"})
        self.assertEqual(result.iterations, 2)

    def test_tie_without_the_current_label_takes_the_lexicographically_smallest(self) -> None:
        # a sits equally between b (weight 2) and c (weight 2): it adopts b, and propagation pulls
        # the whole group behind label b -- but the reported label is then normalised to the minimum
        # member name, which is a.
        graph = edges(("a", "b", 2), ("a", "c", 2))
        result = label_propagation(graph)
        self.assertEqual(result.communities, [["a", "b", "c"]])
        self.assertEqual(result.labels, {"a": "a", "b": "a", "c": "a"})

    def test_labels_are_normalised_to_the_smallest_member_name(self) -> None:
        # Propagation itself ends behind label c (the heavy b-c edge wins b, and a follows), yet the
        # output label must name the smallest member a rather than the label that won the vote.
        graph = edges(("a", "b", 1), ("b", "c", 10))
        result = label_propagation(graph)
        self.assertEqual(result.communities, [["a", "b", "c"]])
        self.assertEqual(set(result.labels), {"a", "b", "c"})
        self.assertTrue(all(value == "a" for value in result.labels.values()))

    def test_directed_mutual_arcs_have_their_weights_added(self) -> None:
        # Directed: a->b and b->a weight 1 each (2 together) against a->c weight 3. Run the same
        # graph both ways: as a directed graph, and as an undirected graph holding one weight-2 arc.
        directed = Graph(directed=True)
        directed.add_edge("a", "b", 1)
        directed.add_edge("b", "a", 1)
        directed.add_edge("a", "c", 3)
        undirected = edges(("a", "b", 2), ("a", "c", 3))
        self.assertEqual(
            label_propagation(directed).to_document(),
            label_propagation(undirected).to_document(),
        )

    def test_self_loops_never_count_as_neighbour_votes(self) -> None:
        # A heavy self-loop on c must not vote for c's own label; the mutual edge is what counts.
        graph = edges(("a", "b", 1), ("b", "b", 100))
        result = label_propagation(graph)
        self.assertEqual(result.communities, [["a", "b"]])

    def test_zero_weight_edges_participate_but_never_pull(self) -> None:
        graph = edges(("a", "b", 0), ("a", "c", 1))
        result = label_propagation(graph)
        self.assertEqual(result.communities, [["a", "b", "c"]])

    def test_same_graph_is_identical_when_edge_input_order_changes(self) -> None:
        rows = [("a", "b", 2), ("c", "b", 3), ("a", "c", 1), ("d", "e", 4), ("c", "d", 1), ("a", "a", 9)]

        def built(order: list[tuple[str, str, float]]) -> CommunityResult:
            graph = Graph(directed=True)
            for source, target, weight in order:
                graph.add_edge(source, target, weight)
            return label_propagation(graph)

        self.assertEqual(built(rows).to_document(), built(list(reversed(rows))).to_document())

    def test_reaching_the_cap_returns_a_deterministic_unconverged_result(self) -> None:
        # A two-cluster graph asked to run a single round cannot have finished: a changes in round 1
        # and a whole no-change round never happened. The labels as they stand are still grouped.
        graph = edges(
            ("a", "b", 3), ("a", "c", 2), ("b", "c", 2),
            ("c", "d", 1),
            ("d", "e", 2), ("d", "f", 2), ("e", "f", 3),
        )
        result = label_propagation(graph, max_iterations=1)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.count, 3)
        self.assertEqual(result.communities, [["a", "b", "c"], ["d"], ["e", "f"]])
        self.assertEqual(result.labels, {"a": "a", "b": "a", "c": "a", "d": "d", "e": "e", "f": "e"})
        # The capped result is itself deterministic.
        self.assertEqual(result.to_document(), label_propagation(graph, max_iterations=1).to_document())

    def test_negative_weight_is_refused_with_the_first_stable_edge(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("x", "y", 1)
        graph.add_edge("b", "c", -2)
        graph.add_edge("a", "d", -5)
        with self.assertRaises(NegativeWeightError) as caught:
            label_propagation(graph)
        # Stable edge order sorts sources, so among the negatives a->d is reported first.
        self.assertEqual(caught.exception.context, {"edge": "a->d", "weight": -5.0})
        self.assertEqual(caught.exception.kind, "negative_weight_error")

    def test_undirected_negative_edge_is_reported_in_its_canonical_direction(self) -> None:
        graph = edges(("z", "a", -3))
        with self.assertRaises(NegativeWeightError) as caught:
            label_propagation(graph)
        self.assertEqual(caught.exception.context, {"edge": "a->z", "weight": -3.0})

    def test_negative_self_loop_is_also_rejected(self) -> None:
        graph = edges(("a", "a", -1))
        with self.assertRaises(NegativeWeightError):
            label_propagation(graph)

    def test_max_iterations_validation(self) -> None:
        graph = edges(("a", "b", 1))
        for bad in (0, -1, 1.5, "10", None, [10], True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    label_propagation(graph, max_iterations=bad)  # type: ignore[arg-type]
        for good in (2, 3, 1000):
            with self.subTest(good=good):
                self.assertTrue(label_propagation(graph, max_iterations=good).converged)


class CommunitiesCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_edges(self, rows: list[dict], name: str = "edges.jsonl") -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    ROWS = [
        {"source": "a", "target": "b", "weight": 3},
        {"source": "a", "target": "c", "weight": 2},
        {"source": "b", "target": "c", "weight": 2},
        {"source": "c", "target": "d", "weight": 1},
        {"source": "d", "target": "e", "weight": 2},
        {"source": "d", "target": "f", "weight": 2},
        {"source": "e", "target": "f", "weight": 3},
    ]

    def test_communities_document_shape_and_exit_code(self) -> None:
        path = self.write_edges(self.ROWS)
        code, out, err = self.run_cli(["communities", "--edges", path])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertEqual(
            document,
            {
                "communities": [["a", "b", "c"], ["d", "e", "f"]],
                "labels": {"a": "a", "b": "a", "c": "a", "d": "d", "e": "d", "f": "d"},
                "count": 2,
                "iterations": 3,
                "converged": True,
            },
        )
        self.assertEqual(out.count("\n"), 1)

    def test_describe_publishes_communities_alongside_the_existing_commands(self) -> None:
        code, out, err = self.run_cli(["describe"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        commands = json.loads(out)["subcommands"]
        self.assertIn("communities", commands)
        # The previous thirteen commands all remain, in their existing positions.
        for command in (
            "astar", "bellman-ford", "bfs", "centrality", "clustering", "compare",
            "components", "describe", "dijkstra", "dfs", "pagerank", "stats", "toposort",
        ):
            self.assertIn(command, commands)
        self.assertEqual(len(commands), 14)

    def test_shuffled_edge_input_is_byte_identical_output(self) -> None:
        import random

        rows = [
            {"source": "a", "target": "b", "weight": 2},
            {"source": "c", "target": "b"},
            {"source": "b", "target": "a", "weight": 1},
            {"source": "a", "target": "a", "weight": 5},
            {"source": "c", "target": "d", "weight": -0.0},
            {"source": "d", "target": "e", "weight": 4},
            {"source": "f", "target": "e", "weight": 3},
            {"source": "z", "target": "z"},
        ]
        shuffled = rows[:]
        random.Random(42).shuffle(shuffled)
        first = self.write_edges(rows, "first.jsonl")
        second = self.write_edges(shuffled, "second.jsonl")
        code_a, out_a, err_a = self.run_cli(["communities", "--edges", first, "--directed"])
        code_b, out_b, err_b = self.run_cli(["communities", "--edges", second, "--directed"])
        self.assertEqual((code_a, err_a, code_b, err_b), (EXIT_OK, "", EXIT_OK, ""))
        self.assertEqual(out_a, out_b)

    def test_empty_graph_writes_the_zero_document_and_still_exits_zero(self) -> None:
        path = os.path.join(self.directory.name, "empty.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("# nothing here\n\n")
        code, out, err = self.run_cli(["communities", "--edges", path])
        # Convergence (including the vacuous convergence of the empty graph) exits 0; code 3 is
        # reserved for the iteration cap, not for emptiness.
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertEqual(
            json.loads(out),
            {"communities": [], "count": 0, "converged": True, "iterations": 0, "labels": {}},
        )

    def test_cap_hit_is_stdout_result_with_code_three_not_an_input_error(self) -> None:
        path = self.write_edges(self.ROWS)
        code, out, err = self.run_cli(["communities", "--edges", path, "--max-iterations", "1"])
        self.assertEqual(code, EXIT_NEGATIVE)
        self.assertEqual(err, "")  # not wrapped as a code-2 error document
        document = json.loads(out)
        self.assertFalse(document["converged"])
        self.assertEqual(document["iterations"], 1)
        self.assertEqual(document["count"], 3)

    def test_negative_weight_is_a_code_two_stderr_document_with_nothing_on_stdout(self) -> None:
        path = self.write_edges(
            [{"source": "x", "target": "y", "weight": 1}, {"source": "b", "target": "c", "weight": -2}]
        )
        code, out, err = self.run_cli(["communities", "--edges", path, "--directed"])
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        self.assertEqual(
            json.loads(err),
            {
                "error": "negative_weight_error",
                "message": "label propagation requires non-negative edge weights",
                "edge": "b->c",
                "weight": -2.0,
            },
        )
        self.assertEqual(err.count("\n"), 1)

    def test_bad_max_iterations_value_is_a_code_two_error(self) -> None:
        path = self.write_edges([{"source": "a", "target": "b"}])
        code, out, err = self.run_cli(["communities", "--edges", path, "--max-iterations", "0"])
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_communities_reads_stdin(self) -> None:
        text = "".join(json.dumps(row) + "\n" for row in self.ROWS)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), unittest.mock.patch.object(
            sys, "stdin", io.StringIO(text)
        ):
            code = main(["communities", "--edges", "-"])
        self.assertEqual((code, err.getvalue()), (EXIT_OK, ""))
        self.assertEqual(json.loads(out.getvalue())["count"], 2)


if __name__ == "__main__":
    unittest.main()
