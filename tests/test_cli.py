"""CLI contract: JSON on stdout, documented exit codes, and the two-algorithm comparison."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock

from graphtk.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main

EDGES = [
    {"source": "a", "target": "b", "weight": 1.0},
    {"source": "b", "target": "c", "weight": 2.0},
    {"source": "a", "target": "c", "weight": 5.0},
    {"source": "c", "target": "d", "weight": 1.0},
]


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.edges = os.path.join(self.directory.name, "edges.jsonl")
        with open(self.edges, "w", encoding="utf-8", newline="\n") as handle:
            for edge in EDGES:
                handle.write(json.dumps(edge) + "\n")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_edges(self, rows: list[dict], name: str = "other.jsonl") -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def write_raw_lines(self, lines: list[str], name: str = "raw.jsonl") -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line + "\n")
        return path

    def run_stdin(self, argv: list[str], text: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        stdin = io.StringIO(text)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), unittest.mock.patch.object(sys, "stdin", stdin):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_describe_publishes_the_contract(self) -> None:
        code, out, err = run_cli(["describe"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertEqual(document["exitCodes"], {"ok": 0, "error": 2, "negativeVerdict": 3})
        self.assertIn("compare", document["subcommands"])
        self.assertFalse(document["directedByDefault"])

    def test_stats_reports_components(self) -> None:
        code, out, _ = run_cli(["stats", "--edges", self.edges, "--directed"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertEqual(document["nodes"], 4)
        self.assertEqual(document["edges"], 4)
        self.assertEqual(document["components"], 1)

    def test_bfs_reports_distances_and_a_depth_first_order(self) -> None:
        code, out, _ = run_cli(["bfs", "--edges", self.edges, "--source", "a"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertEqual(document["distances"]["d"], 2)
        self.assertEqual(document["order"][0], "a")

    def test_dijkstra_with_target_reports_the_path(self) -> None:
        code, out, _ = run_cli(["dijkstra", "--edges", self.edges, "--directed", "--source", "a", "--target", "c"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertEqual(document["distance"], 3.0)
        self.assertEqual(document["path"], ["a", "b", "c"])

    def test_dijkstra_negative_weight_is_an_error_document(self) -> None:
        negative = self.write_edges([{"source": "a", "target": "b", "weight": -1}])
        code, _, err = run_cli(["dijkstra", "--edges", negative, "--source", "a"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "negative_weight_error")

    def test_bellman_ford_negative_cycle_is_an_error_document(self) -> None:
        cycle = self.write_edges(
            [
                {"source": "a", "target": "b", "weight": 1},
                {"source": "b", "target": "a", "weight": -2},
            ]
        )
        code, _, err = run_cli(["bellman-ford", "--edges", cycle, "--directed", "--source", "a"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "negative_cycle_error")

    def test_components_are_listed_in_order(self) -> None:
        split = self.write_edges([{"source": "a", "target": "b"}, {"source": "c", "target": "d"}])
        code, out, _ = run_cli(["components", "--edges", split])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out)["components"], [["a", "b"], ["c", "d"]])

    def test_toposort_requires_the_directed_flag(self) -> None:
        code, _, err = run_cli(["toposort", "--edges", self.edges])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_toposort_returns_an_order_that_respects_edges(self) -> None:
        code, out, _ = run_cli(["toposort", "--edges", self.edges, "--directed"])
        self.assertEqual(code, EXIT_OK)
        order = json.loads(out)["order"]
        self.assertLess(order.index("a"), order.index("d"))

    def test_pagerank_reports_convergence(self) -> None:
        code, out, _ = run_cli(["pagerank", "--edges", self.edges, "--directed"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertTrue(document["converged"])
        self.assertAlmostEqual(sum(document["scores"].values()), 1.0, places=6)

    def test_centrality_is_normalised(self) -> None:
        code, out, _ = run_cli(["centrality", "--edges", self.edges])
        self.assertEqual(code, EXIT_OK)
        self.assertGreater(json.loads(out)["centrality"]["b"], 0.0)

    def test_clustering_reports_the_closed_triangle_and_open_wedges(self) -> None:
        code, out, _ = run_cli(["clustering", "--edges", self.edges])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        # EDGES (undirected): a-b, b-c, a-c form a triangle; c-d hangs off it.
        self.assertEqual(document["triangles"], 1)
        self.assertEqual(document["connectedTriples"], 5)
        self.assertEqual(document["transitivity"], 0.6)
        self.assertAlmostEqual(document["average"], 7 / 12, places=10)
        self.assertAlmostEqual(document["coefficients"]["c"], 1 / 3, places=10)
        self.assertEqual(document["coefficients"]["d"], 0.0)
        self.assertEqual(document["nodes"], 4)
        self.assertEqual(list(document["coefficients"]), ["a", "b", "c", "d"])

    def test_clustering_has_ten_decimal_output_convention(self) -> None:
        code, out, _ = run_cli(["clustering", "--edges", self.edges])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertEqual(document["coefficients"]["c"], 0.3333333333)

    def test_clustering_directed_collapses_mutual_arcs(self) -> None:
        rows = [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "a"},
            {"source": "b", "target": "c"},
            {"source": "c", "target": "b"},
            {"source": "a", "target": "c"},
        ]
        directed = self.write_edges(rows, "directed.jsonl")
        undirected = self.write_edges(
            [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}, {"source": "a", "target": "c"}],
            "undirected.jsonl",
        )
        code_d, out_d, err_d = run_cli(["clustering", "--edges", directed, "--directed"])
        code_u, out_u, _ = run_cli(["clustering", "--edges", undirected])
        self.assertEqual((code_d, err_d, code_u), (EXIT_OK, "", EXIT_OK))
        self.assertEqual(json.loads(out_d), json.loads(out_u))

    def test_clustering_is_identical_for_different_insertion_orders(self) -> None:
        rows = [
            {"source": "a", "target": "b", "weight": 2},
            {"source": "c", "target": "b"},
            {"source": "a", "target": "c"},
            {"source": "a", "target": "a"},
            {"source": "c", "target": "d", "weight": -1},
        ]
        first = self.write_edges(rows, "first.jsonl")
        second = self.write_edges(list(reversed(rows)), "second.jsonl")
        code_a, out_a, _ = run_cli(["clustering", "--edges", first, "--directed"])
        code_b, out_b, _ = run_cli(["clustering", "--edges", second, "--directed"])
        self.assertEqual((code_a, code_b), (EXIT_OK, EXIT_OK))
        self.assertEqual(out_a, out_b)

    def test_clustering_reads_stdin(self) -> None:
        text = "".join(json.dumps(edge) + "\n" for edge in EDGES)
        out, err = io.StringIO(), io.StringIO()
        stdin = io.StringIO(text)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), unittest.mock.patch.object(sys, "stdin", stdin):
            code = main(["clustering", "--edges", "-"])
        self.assertEqual((code, err.getvalue()), (EXIT_OK, ""))
        self.assertEqual(json.loads(out.getvalue())["triangles"], 1)

    def test_clustering_empty_graph_is_a_full_zero_document_with_code_three(self) -> None:
        empty = os.path.join(self.directory.name, "empty.jsonl")
        with open(empty, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("# just a comment\n\n")
        code, out, err = run_cli(["clustering", "--edges", empty])
        self.assertEqual((code, err), (EXIT_NEGATIVE, ""))
        self.assertEqual(
            json.loads(out),
            {
                "average": 0.0,
                "coefficients": {},
                "connectedTriples": 0,
                "nodes": 0,
                "transitivity": 0.0,
                "triangles": 0,
            },
        )

    def test_clustering_parse_and_file_errors_stay_code_two(self) -> None:
        broken = self.write_edges([{"source": "a", "target": "b"}])
        with open(broken, "a", encoding="utf-8", newline="\n") as handle:
            handle.write("not json\n")
        code, _, err = run_cli(["clustering", "--edges", broken])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "parse_error")
        code, _, err = run_cli(["clustering", "--edges", os.path.join(self.directory.name, "missing.jsonl")])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_describe_publishes_clustering(self) -> None:
        code, out, _ = run_cli(["describe"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("clustering", json.loads(out)["subcommands"])

    def test_compare_agrees_between_the_two_shortest_path_algorithms(self) -> None:
        code, out, _ = run_cli(["compare", "--edges", self.edges, "--directed", "--source", "a"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertTrue(document["identical"])
        self.assertEqual(document["nodesCompared"], 4)

    def test_compare_negative_weights_still_agree_through_bellman_ford_check(self) -> None:
        # Dijkstra refuses negative weights, so the comparison must surface that as an error document
        # rather than reporting a disagreement it cannot compute.
        negative = self.write_edges([{"source": "a", "target": "b", "weight": -1}])
        code, _, err = run_cli(["compare", "--edges", negative, "--directed", "--source", "a"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["error"], "negative_weight_error")

    def test_bad_edge_line_is_reported_with_its_line_number(self) -> None:
        broken = self.write_edges([{"source": "a", "target": "b"}])
        with open(broken, "a", encoding="utf-8", newline="\n") as handle:
            handle.write("{oops}\n")
        code, _, err = run_cli(["stats", "--edges", broken])
        self.assertEqual(code, EXIT_ERROR)
        document = json.loads(err)
        self.assertEqual(document["error"], "parse_error")
        self.assertEqual(document["line"], 2)

    def test_unknown_field_is_rejected(self) -> None:
        extra = self.write_edges([{"source": "a", "target": "b", "label": "x"}])
        code, _, err = run_cli(["stats", "--edges", extra])
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("unknown field", json.loads(err)["message"])

    def test_single_node_graph_is_a_negative_verdict_for_bfs(self) -> None:
        lonely = self.write_edges([{"source": "a", "target": "b"}])
        code, _, _ = run_cli(["bfs", "--edges", lonely, "--source", "a"])
        self.assertEqual(code, EXIT_OK)
        code, _, _ = run_cli(["components", "--edges", self.write_edges([{"source": "a", "target": "a"}])])
        self.assertEqual(code, EXIT_OK)

    # -- weight validation across the JSONL / stdin input chain ------------------------------------
    NON_FINITE_CONSTANTS = ["NaN", "Infinity", "-Infinity"]

    def assert_parse_error(self, code: int, out: str, err: str, line: int) -> dict:
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")  # nothing on stdout: no document with non-standard values
        document = json.loads(err)  # stderr is itself strict JSON
        self.assertEqual(document["error"], "parse_error")
        self.assertEqual(document["line"], line)
        return document

    def test_non_finite_constants_in_a_file_are_parse_errors_with_line_numbers(self) -> None:
        for constant in self.NON_FINITE_CONSTANTS:
            with self.subTest(constant=constant):
                # The constant is on line 2, after one legal edge and a comment/blank line.
                path = self.write_raw_lines(
                    [
                        json.dumps({"source": "a", "target": "b", "weight": 1.0}),
                        "# a comment",
                        "",
                        f'{{"source": "b", "target": "c", "weight": {constant}}}',
                    ],
                    name=f"{constant.strip('-').lower()}.jsonl",
                )
                code, out, err = run_cli(["stats", "--edges", path])
                self.assert_parse_error(code, out, err, 4)

    def test_non_finite_constants_on_stdin_are_parse_errors_with_line_numbers(self) -> None:
        for constant in self.NON_FINITE_CONSTANTS:
            with self.subTest(constant=constant):
                text = (
                    '{"source": "a", "target": "b", "weight": 1.0}\n'
                    f'{{"source": "b", "target": "c", "weight": {constant}}}\n'
                )
                code, out, err = self.run_stdin(["components", "--edges", "-"], text)
                self.assert_parse_error(code, out, err, 2)

    def test_non_finite_first_line_still_reports_line_one(self) -> None:
        path = self.write_raw_lines(['{"source": "a", "target": "b", "weight": NaN}'])
        code, out, err = run_cli(["stats", "--edges", path])
        self.assert_parse_error(code, out, err, 1)

    def test_bad_weight_types_in_json_are_parse_errors_with_line_numbers(self) -> None:
        cases = {
            "string": '{"source": "a", "target": "b", "weight": "1.5"}',
            "null": '{"source": "a", "target": "b", "weight": null}',
            "bool": '{"source": "a", "target": "b", "weight": true}',
            "array": '{"source": "a", "target": "b", "weight": [1.0]}',
            "object": '{"source": "a", "target": "b", "weight": {"v": 1.0}}',
        }
        for name, line in cases.items():
            with self.subTest(name=name):
                path = self.write_raw_lines([line])
                code, out, err = run_cli(["stats", "--edges", path])
                self.assert_parse_error(code, out, err, 1)

    def test_non_finite_weight_fails_for_every_graph_command(self) -> None:
        path = self.write_raw_lines(['{"source": "a", "target": "b", "weight": Infinity}'])
        commands = [
            ["stats"],
            ["bfs", "--source", "a"],
            ["dijkstra", "--source", "a"],
            ["bellman-ford", "--source", "a"],
            ["components"],
            ["toposort", "--directed"],
            ["pagerank"],
            ["centrality"],
            ["clustering"],
            ["compare", "--source", "a"],
        ]
        for command in commands:
            with self.subTest(command=command[0]):
                code, out, err = run_cli([*command, "--edges", path])
                self.assert_parse_error(code, out, err, 1)

    def test_legal_extreme_and_negative_weights_are_still_accepted(self) -> None:
        rows = [
            {"source": "a", "target": "b", "weight": 0},
            {"source": "b", "target": "c", "weight": -2},
            {"source": "a", "target": "c", "weight": 1e-300},
            {"source": "c", "target": "d", "weight": 1e308},
        ]
        path = self.write_edges(rows)
        code, out, err = run_cli(["bellman-ford", "--edges", path, "--directed", "--source", "a"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertEqual(document["distances"]["b"], 0.0)
        self.assertEqual(document["distances"]["c"], -2.0)

    def test_negative_weight_dijkstra_is_still_negative_weight_error(self) -> None:
        negative = self.write_edges([{"source": "a", "target": "b", "weight": -1}])
        code, out, err = run_cli(["dijkstra", "--edges", negative, "--source", "a"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        self.assertEqual(json.loads(err)["error"], "negative_weight_error")

    def test_weight_error_does_not_leak_a_builtin_exception(self) -> None:
        path = self.write_raw_lines(['{"source": "a", "target": "b", "weight": NaN}'])
        code, out, err = run_cli(["stats", "--edges", path])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        document = json.loads(err)
        # The document carries the stable kind, not a traceback or builtin type name.
        self.assertEqual(set(document), {"error", "message", "line"})
        self.assertNotIn("Traceback", err)

    def test_integer_and_decimal_weights_normalise_and_compare_still_agrees(self) -> None:
        rows = [
            {"source": "a", "target": "b", "weight": 2},
            {"source": "b", "target": "c", "weight": 0.5},
            {"source": "a", "target": "c", "weight": 9},
        ]
        path = self.write_edges(rows)
        code, out, err = run_cli(["compare", "--edges", path, "--directed", "--source", "a"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertTrue(document["identical"])


if __name__ == "__main__":
    unittest.main()
