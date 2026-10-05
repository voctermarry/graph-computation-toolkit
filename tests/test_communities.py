"""Label-propagation communities: determinism, tie-breaking, validation, and the CLI contract."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from graphtk import (
    CommunitiesResult,
    Edge,
    Graph,
    NegativeWeightError,
    ValidationError,
    label_propagation,
)
from graphtk.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def _edge(row: dict) -> Edge:
    return Edge(row["source"], row["target"], row.get("weight", 1.0))


def two_cluster_edges() -> list[dict]:
    """Two dense triangles joined by one weak bridge: a-b-c and d-e-f."""
    return [
        {"source": "a", "target": "b", "weight": 5.0},
        {"source": "b", "target": "c", "weight": 5.0},
        {"source": "a", "target": "c", "weight": 5.0},
        {"source": "d", "target": "e", "weight": 5.0},
        {"source": "e", "target": "f", "weight": 5.0},
        {"source": "d", "target": "f", "weight": 5.0},
        {"source": "c", "target": "d", "weight": 0.5},
    ]


class LabelPropagationTests(unittest.TestCase):
    def test_two_clusters_are_separated(self) -> None:
        graph = Graph.from_edges([_edge(row) for row in two_cluster_edges()])
        result = label_propagation(graph)
        self.assertTrue(result.converged)
        self.assertEqual(result.communities, [["a", "b", "c"], ["d", "e", "f"]])
        self.assertEqual(result.count, 2)
        self.assertEqual(
            result.labels,
            {"a": "a", "b": "a", "c": "a", "d": "d", "e": "d", "f": "d"},
        )

    def test_document_shape(self) -> None:
        graph = Graph.from_edges([_edge({"source": "a", "target": "b"})])
        document = label_propagation(graph).to_document()
        self.assertEqual(
            document,
            {
                "communities": [["a", "b"]],
                "labels": {"a": "a", "b": "a"},
                "count": 1,
                "iterations": document["iterations"],
                "converged": True,
            },
        )
        self.assertGreaterEqual(document["iterations"], 1)

    def test_empty_graph(self) -> None:
        result = label_propagation(Graph())
        self.assertEqual(result, CommunitiesResult([], {}, 0, True))
        self.assertEqual(
            result.to_document(),
            {"communities": [], "labels": {}, "count": 0, "iterations": 0, "converged": True},
        )

    def test_isolated_nodes_stay_singletons(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b", 2.0)
        graph.add_node("zz")
        result = label_propagation(graph)
        self.assertEqual(result.communities, [["a", "b"], ["zz"]])
        self.assertEqual(result.labels["zz"], "zz")

    def test_self_loops_are_ignored(self) -> None:
        graph = Graph()
        graph.add_edge("a", "a", 9.0)
        graph.add_edge("a", "b", 1.0)
        result = label_propagation(graph)
        self.assertEqual(result.communities, [["a", "b"]])

    def test_directed_mutual_arcs_sum_weights(self) -> None:
        # a->b is weak, b->a is strong: together they outweigh the c-d side for b.
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("b", "a", 4.0)
        graph.add_edge("b", "c", 2.0)
        graph.add_edge("c", "d", 5.0)
        graph.add_edge("d", "c", 5.0)
        result = label_propagation(graph)
        for first, second in (("a", "b"), ("c", "d")):
            self.assertEqual(result.labels[first], result.labels[second])
        self.assertNotEqual(result.labels["a"], result.labels["c"])

    def test_tie_keeps_current_label(self) -> None:
        # Path a-b-c, unit weights: in round one b sees labels "b" (from a, already updated) and
        # "c" tied at 1, and keeps its own label "b"; the whole path settles on it.
        graph = Graph()
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("b", "c", 1.0)
        result = label_propagation(graph)
        self.assertTrue(result.converged)
        self.assertEqual(result.communities, [["a", "b", "c"]])
        self.assertEqual(result.labels, {"a": "a", "b": "a", "c": "a"})

    def test_tie_without_current_label_picks_lexicographic_minimum(self) -> None:
        # m is equally drawn to the "z1" and "z2" camps; its own label is not tied, so it must
        # take the lexicographically smaller of the tied labels.
        graph = Graph()
        graph.add_edge("a", "z1", 10.0)
        graph.add_edge("b", "z2", 10.0)
        graph.add_edge("m", "a", 1.0)
        graph.add_edge("m", "b", 1.0)
        result = label_propagation(graph)
        self.assertTrue(result.converged)
        self.assertEqual(result.communities, [["a", "m", "z1"], ["b", "z2"]])

    def test_iteration_cap_reports_not_converged_but_still_deterministic(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b", 1.0)
        result = label_propagation(graph, max_iterations=1)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.communities, [["a", "b"]])
        again = label_propagation(graph, max_iterations=1)
        self.assertEqual(result.to_document(), again.to_document())

    def test_result_is_deterministic_under_edge_order(self) -> None:
        rows = two_cluster_edges()
        forward = Graph.from_edges([_edge(row) for row in rows])
        shuffled = Graph.from_edges([_edge(row) for row in reversed(rows)])
        self.assertEqual(
            label_propagation(forward).to_document(),
            label_propagation(shuffled).to_document(),
        )

    def test_negative_weight_is_refused_with_first_edge_in_stable_order(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("z", "a", -3.0)
        graph.add_edge("a", "b", -1.0)
        with self.assertRaises(NegativeWeightError) as caught:
            label_propagation(graph)
        self.assertEqual(caught.exception.kind, "negative_weight_error")
        self.assertEqual(caught.exception.context["edge"], "a->b")
        self.assertEqual(caught.exception.context["weight"], -1.0)

    def test_max_iterations_must_be_a_positive_integer(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b")
        for bad in (0, -1, True, 1.5, "3", None):
            with self.subTest(value=bad):
                with self.assertRaises(ValidationError):
                    label_propagation(graph, max_iterations=bad)  # type: ignore[arg-type]


class CommunitiesCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.edges = os.path.join(self.directory.name, "edges.jsonl")
        self.write_rows(two_cluster_edges(), self.edges)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_rows(self, rows: list[dict], path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_converged_run_exits_zero_with_sorted_document(self) -> None:
        code, out, err = run_cli(["communities", "--edges", self.edges])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(err, "")
        document = json.loads(out)
        self.assertEqual(document["communities"], [["a", "b", "c"], ["d", "e", "f"]])
        self.assertEqual(document["count"], 2)
        self.assertTrue(document["converged"])
        self.assertEqual(document["labels"]["f"], "d")

    def test_output_is_byte_identical_under_edge_reordering(self) -> None:
        shuffled = os.path.join(self.directory.name, "shuffled.jsonl")
        self.write_rows(list(reversed(two_cluster_edges())), shuffled)
        code_a, out_a, _ = run_cli(["communities", "--edges", self.edges])
        code_b, out_b, _ = run_cli(["communities", "--edges", shuffled])
        self.assertEqual(code_a, code_b)
        self.assertEqual(out_a, out_b)

    def test_directed_flag_is_accepted(self) -> None:
        code, out, _ = run_cli(["communities", "--edges", self.edges, "--directed"])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out)["count"], 2)

    def test_iteration_cap_exits_three_but_still_writes_the_result(self) -> None:
        code, out, err = run_cli(["communities", "--edges", self.edges, "--max-iterations", "1"])
        self.assertEqual(code, EXIT_NEGATIVE)
        self.assertEqual(err, "")
        document = json.loads(out)
        self.assertFalse(document["converged"])
        self.assertEqual(document["iterations"], 1)
        self.assertEqual(document["count"], len(document["communities"]))

    def test_empty_input_converges_with_zero_count(self) -> None:
        empty = os.path.join(self.directory.name, "empty.jsonl")
        self.write_rows([], empty)
        code, out, _ = run_cli(["communities", "--edges", empty])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(
            json.loads(out),
            {"communities": [], "labels": {}, "count": 0, "iterations": 0, "converged": True},
        )

    def test_negative_weight_is_an_input_error_not_a_verdict(self) -> None:
        bad = os.path.join(self.directory.name, "bad.jsonl")
        self.write_rows([{"source": "a", "target": "b", "weight": -1.0}], bad)
        code, out, err = run_cli(["communities", "--edges", bad])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        document = json.loads(err)
        self.assertEqual(document["error"], "negative_weight_error")
        self.assertEqual(document["edge"], "a->b")
        self.assertEqual(document["weight"], -1.0)

    def test_invalid_max_iterations_is_a_validation_error(self) -> None:
        code, out, err = run_cli(["communities", "--edges", self.edges, "--max-iterations", "0"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_describe_lists_communities(self) -> None:
        code, out, _ = run_cli(["describe"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("communities", json.loads(out)["subcommands"])


if __name__ == "__main__":
    unittest.main()
