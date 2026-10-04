"""A* shortest path: algorithm contract, heuristic validation, CLI and Dijkstra equivalence.

The acceptance promises: directed and undirected graphs, zero-weight edges, unreachable targets,
heuristic validation (finite, non-negative, graph nodes only, zero at the target, consistent),
name-order tie-breaks and byte-deterministic output, and an empty heuristic reproducing Dijkstra's
distances and reconstructed paths exactly.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from graphtk import (
    Graph,
    NegativeWeightError,
    ValidationError,
    astar,
    dijkstra,
    path_from,
)
from graphtk.cli import EXIT_ERROR, EXIT_OK, main


def directed_sample() -> Graph:
    graph = Graph(directed=True)
    graph.add_edge("a", "b", 1.0)
    graph.add_edge("b", "c", 2.0)
    graph.add_edge("a", "c", 5.0)
    graph.add_edge("c", "d", 1.0)
    return graph


class AStarAlgorithmTests(unittest.TestCase):
    def test_directed_graph_finds_the_short_route(self) -> None:
        graph = directed_sample()
        distance, previous, expanded = astar(graph, "a", "d", {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0})
        self.assertEqual(distance["d"], 4.0)
        self.assertEqual(path_from(previous, "d"), ["a", "b", "c", "d"])
        self.assertEqual(expanded, 4)

    def test_expanded_counts_distinct_nodes_and_a_good_heuristic_prunes(self) -> None:
        # The expensive branch through x is never expanded once the heuristic prices it correctly.
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("a", "x", 100.0)
        graph.add_edge("b", "d", 1.0)
        graph.add_edge("x", "d", 100.0)
        _, _, expanded = astar(graph, "a", "d", {"a": 2.0, "b": 1.0, "x": 100.0, "d": 0.0})
        self.assertEqual(expanded, 3)

    def test_undirected_graph_walks_both_directions(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("b", "c", 1.0)
        distance, previous, expanded = astar(graph, "c", "a", {"a": 0.0, "b": 1.0, "c": 2.0})
        self.assertEqual(distance["a"], 2.0)
        self.assertEqual(path_from(previous, "a"), ["c", "b", "a"])
        self.assertEqual(expanded, 3)

    def test_zero_weight_edges_tie_break_by_node_name(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 0.0)
        graph.add_edge("a", "c", 0.0)
        graph.add_edge("b", "c", 0.0)
        distance, previous, expanded = astar(graph, "a", "c", {"a": 0.0, "b": 0.0, "c": 0.0})
        self.assertEqual(distance["c"], 0.0)
        # Equal-cost routes: the direct lexicographically first predecessor wins, exactly like Dijkstra.
        self.assertEqual(path_from(previous, "c"), ["a", "c"])
        self.assertEqual(expanded, 3)

    def test_equal_cost_diamond_prefers_the_named_predecessor(self) -> None:
        graph = Graph(directed=True)
        for source, target in (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")):
            graph.add_edge(source, target, 1.0)
        _, previous, expanded = astar(graph, "a", "d", {})
        self.assertEqual(path_from(previous, "d"), ["a", "b", "d"])
        self.assertEqual(expanded, 4)

    def test_source_equals_target(self) -> None:
        distance, previous, expanded = astar(directed_sample(), "a", "a", {"a": 0.0})
        self.assertEqual(distance, {"a": 0.0})
        self.assertEqual(path_from(previous, "a"), ["a"])
        self.assertEqual(expanded, 1)

    def test_unknown_source_and_target_are_validation_errors(self) -> None:
        with self.assertRaises(ValidationError):
            astar(directed_sample(), "nope", "d", {})
        with self.assertRaises(ValidationError):
            astar(directed_sample(), "a", "nope", {})

    def test_unreachable_target_names_the_target(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1.0)
        graph.add_node("z")
        with self.assertRaises(ValidationError) as caught:
            astar(graph, "a", "z", {})
        self.assertIn("unreachable target", caught.exception.message)
        self.assertEqual(caught.exception.context["target"], "z")

    def test_negative_weight_is_refused_with_the_edge(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("b", "c", -3.0)
        with self.assertRaises(NegativeWeightError) as caught:
            astar(graph, "a", "c", {})
        self.assertEqual(caught.exception.context["edge"], "b->c")
        self.assertEqual(caught.exception.context["weight"], -3.0)

    def test_heuristic_value_types_are_validated(self) -> None:
        graph = directed_sample()
        for bad in (True, False, "1.0", None, float("nan"), float("inf"), float("-inf"), -0.5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    astar(graph, "a", "d", {"b": bad})  # type: ignore[dict-item]

    def test_heuristic_rejects_unknown_nodes(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            astar(directed_sample(), "a", "d", {"ghost": 1.0})
        self.assertEqual(caught.exception.context["value"], "ghost")

    def test_heuristic_at_target_must_be_zero(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            astar(directed_sample(), "a", "d", {"d": 0.5})
        self.assertEqual(caught.exception.context["node"], "d")

    def test_inconsistent_heuristic_reports_first_edge_in_node_order(self) -> None:
        graph = directed_sample()
        # h(a)=9 overshoots the a-b arc; a->b sorts before the equally violating a->c.
        with self.assertRaises(ValidationError) as caught:
            astar(graph, "a", "d", {"a": 9.0, "b": 0.0, "c": 0.0, "d": 0.0})
        self.assertEqual(caught.exception.context["edge"], "a->b")
        # A later violation is still caught once the first one is repaired.
        with self.assertRaises(ValidationError) as caught:
            astar(graph, "a", "d", {"a": 1.0, "b": 4.0, "c": 0.0, "d": 0.0})
        self.assertEqual(caught.exception.context["edge"], "b->c")

    def test_validation_error_is_stable_across_mapping_order(self) -> None:
        graph = directed_sample()
        first = {"a": 9.0, "b": 9.0, "d": 0.0}
        second = {"d": 0.0, "b": 9.0, "a": 9.0}
        with self.assertRaises(ValidationError) as caught_first:
            astar(graph, "a", "d", first)
        with self.assertRaises(ValidationError) as caught_second:
            astar(graph, "a", "d", second)
        self.assertEqual(caught_first.exception.context, caught_second.exception.context)

    def test_empty_heuristic_matches_dijkstra_distances_and_paths(self) -> None:
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
        dj_distance, dj_previous = dijkstra(graph, "a")
        for target in ("b", "c", "d", "e"):
            with self.subTest(target=target):
                a_distance, a_previous, _ = astar(graph, "a", target, {})
                self.assertEqual(a_distance[target], dj_distance[target])
                self.assertEqual(path_from(a_previous, target), path_from(dj_previous, target))
        with self.assertRaises(ValidationError):
            astar(graph, "a", "island", {})

    def test_empty_heuristic_matches_dijkstra_on_undirected_and_zero_weights(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b", 0.0)
        graph.add_edge("b", "c", 0.0)
        graph.add_edge("a", "c", 4.0)
        graph.add_edge("c", "d", 2.5)
        dj_distance, dj_previous = dijkstra(graph, "a")
        a_distance, a_previous, _ = astar(graph, "a", "d", {})
        self.assertEqual(a_distance, dj_distance)
        self.assertEqual(path_from(a_previous, "d"), path_from(dj_previous, "d"))

    def test_consistent_heuristic_agrees_with_dijkstra(self) -> None:
        graph = directed_sample()
        heuristic = {"a": 3.0, "b": 3.0, "c": 1.0, "d": 0.0}
        a_distance, a_previous, _ = astar(graph, "a", "d", heuristic)
        dj_distance, dj_previous = dijkstra(graph, "a")
        self.assertEqual(a_distance["d"], dj_distance["d"])
        self.assertEqual(path_from(a_previous, "d"), path_from(dj_previous, "d"))

    def test_result_is_deterministic(self) -> None:
        graph = directed_sample()
        heuristic = {"d": 0.0, "c": 1.0, "a": 3.0, "b": 2.0}
        first = astar(graph, "a", "d", heuristic)
        second = astar(graph, "a", "d", dict(reversed(list(heuristic.items()))))
        self.assertEqual(first, second)
        self.assertIsInstance(first[0]["d"], float)


class AStarCLITests(unittest.TestCase):
    EDGES = [
        {"source": "a", "target": "b", "weight": 1.0},
        {"source": "b", "target": "c", "weight": 2.0},
        {"source": "a", "target": "c", "weight": 5.0},
        {"source": "c", "target": "d", "weight": 1.0},
    ]

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.edges = os.path.join(self.directory.name, "edges.jsonl")
        with open(self.edges, "w", encoding="utf-8", newline="\n") as handle:
            for edge in self.EDGES:
                handle.write(json.dumps(edge) + "\n")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_edges(self, rows: list[dict], name: str = "other.jsonl") -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def write_heuristic(self, text: str, name: str = "heuristic.json") -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return path

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_success_document(self) -> None:
        heuristic = self.write_heuristic(json.dumps({"a": 3, "b": 2, "c": 1, "d": 0}))
        code, out, err = self.run_cli(
            ["astar", "--edges", self.edges, "--directed", "--source", "a", "--target", "d", "--heuristic", heuristic]
        )
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertEqual(
            json.loads(out),
            {"source": "a", "target": "d", "distance": 4.0, "path": ["a", "b", "c", "d"], "expanded": 4},
        )

    def test_output_is_byte_stable_across_edge_and_key_order(self) -> None:
        reordered = self.write_edges(list(reversed(self.EDGES)), "reordered.jsonl")
        first = self.write_heuristic('{"a": 3, "b": 2, "c": 1, "d": 0}', "h1.json")
        second = self.write_heuristic('{\n  "d": 0,\n  "c": 1,\n  "b": 2,\n  "a": 3\n}\n', "h2.json")
        argv = ["astar", "--edges", self.edges, "--directed", "--source", "a", "--target", "d", "--heuristic"]
        _, out_one, _ = self.run_cli([*argv, first])
        _, out_two, _ = self.run_cli(["astar", "--edges", reordered, "--directed", "--source", "a", "--target", "d", "--heuristic", second])
        self.assertEqual(out_one, out_two)

    def test_empty_heuristic_matches_dijkstra_command(self) -> None:
        heuristic = self.write_heuristic("{}\n")
        code, out, err = self.run_cli(
            ["astar", "--edges", self.edges, "--directed", "--source", "a", "--target", "d", "--heuristic", heuristic]
        )
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertEqual((document["distance"], document["path"]), (4.0, ["a", "b", "c", "d"]))
        self.assertEqual(document["expanded"], 4)
        _, dijkstra_out, _ = self.run_cli(
            ["dijkstra", "--edges", self.edges, "--directed", "--source", "a", "--target", "d"]
        )
        self.assertEqual(document["distance"], json.loads(dijkstra_out)["distance"])
        self.assertEqual(document["path"], json.loads(dijkstra_out)["path"])

    def test_undirected_and_zero_weight_edges(self) -> None:
        rows = [
            {"source": "a", "target": "b", "weight": 0},
            {"source": "a", "target": "c", "weight": 0},
            {"source": "b", "target": "c", "weight": 0},
        ]
        edges = self.write_edges(rows)
        heuristic = self.write_heuristic(json.dumps({"a": 0, "b": 0, "c": 0}))
        code, out, err = self.run_cli(
            ["astar", "--edges", edges, "--source", "a", "--target", "c", "--heuristic", heuristic]
        )
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertEqual((document["distance"], document["path"]), (0.0, ["a", "c"]))

    def test_unreachable_target_is_code_two_with_empty_stdout(self) -> None:
        edges = self.write_edges(
            [{"source": "a", "target": "b", "weight": 1.0}, {"source": "z", "target": "y", "weight": 1.0}]
        )
        heuristic = self.write_heuristic("{}")
        code, out, err = self.run_cli(
            ["astar", "--edges", edges, "--directed", "--source", "a", "--target", "z", "--heuristic", heuristic]
        )
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertIn("unreachable target", document["message"])

    def test_unknown_endpoints_are_validation_errors(self) -> None:
        heuristic = self.write_heuristic("{}")
        base = ["astar", "--edges", self.edges, "--directed", "--heuristic", heuristic]
        code, out, err = self.run_cli([*base, "--source", "nope", "--target", "d"])
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        self.assertEqual(json.loads(err)["error"], "validation_error")
        code, _, err = self.run_cli([*base, "--source", "a", "--target", "nope"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_negative_weight_is_an_error_document(self) -> None:
        edges = self.write_edges([{"source": "a", "target": "b", "weight": -1.0}])
        heuristic = self.write_heuristic("{}")
        code, out, err = self.run_cli(
            ["astar", "--edges", edges, "--directed", "--source", "a", "--target", "b", "--heuristic", heuristic]
        )
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        document = json.loads(err)
        self.assertEqual(document["error"], "negative_weight_error")
        self.assertEqual(document["edge"], "a->b")
        self.assertEqual(document["weight"], -1.0)

    def test_bad_heuristic_files_are_parse_errors_with_no_partial_output(self) -> None:
        cases = {
            "not json": "not json\n",
            "array": "[1, 2, 3]",
            "null": "null",
            "number": "42",
            "boolean value": '{"b": true}',
            "string value": '{"b": "1"}',
            "null value": '{"b": null}',
            "nan constant": '{"b": NaN}',
            "infinity constant": '{"b": Infinity}',
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                heuristic = self.write_heuristic(text, name=f"{name.replace(' ', '_')}.json")
                code, out, err = self.run_cli(
                    [
                        "astar",
                        "--edges",
                        self.edges,
                        "--directed",
                        "--source",
                        "a",
                        "--target",
                        "d",
                        "--heuristic",
                        heuristic,
                    ]
                )
                self.assertEqual((code, out), (EXIT_ERROR, ""))
                self.assertEqual(json.loads(err)["error"], "parse_error")

    def test_heuristic_validation_failures_are_code_two(self) -> None:
        cases = {
            "unknown node": json.dumps({"ghost": 1.0, "d": 0.0}),
            "target nonzero": json.dumps({"d": 0.5}),
            "inconsistent": json.dumps({"a": 9.0, "d": 0.0}),
            "negative value": json.dumps({"b": -0.1, "d": 0.0}),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                heuristic = self.write_heuristic(text, name=f"{name.replace(' ', '_')}.json")
                code, out, err = self.run_cli(
                    [
                        "astar",
                        "--edges",
                        self.edges,
                        "--directed",
                        "--source",
                        "a",
                        "--target",
                        "d",
                        "--heuristic",
                        heuristic,
                    ]
                )
                self.assertEqual((code, out), (EXIT_ERROR, ""))
                self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_inconsistency_error_names_first_edge_in_node_order(self) -> None:
        # h(a)=9 violates both a->b (9 > 1) and a->c (9 > 5); a->b sorts first.
        heuristic = self.write_heuristic(json.dumps({"a": 9.0, "b": 0.0, "c": 0.0, "d": 0.0}))
        code, _, err = self.run_cli(
            ["astar", "--edges", self.edges, "--directed", "--source", "a", "--target", "d", "--heuristic", heuristic]
        )
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["edge"], "a->b")

    def test_missing_heuristic_file_is_an_error(self) -> None:
        code, out, err = self.run_cli(
            [
                "astar",
                "--edges",
                self.edges,
                "--directed",
                "--source",
                "a",
                "--target",
                "d",
                "--heuristic",
                os.path.join(self.directory.name, "missing.json"),
            ]
        )
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_describe_lists_astar(self) -> None:
        code, out, err = self.run_cli(["describe"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertIn("astar", json.loads(out)["subcommands"])


if __name__ == "__main__":
    unittest.main()
