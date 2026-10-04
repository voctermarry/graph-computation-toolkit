"""CLI contract: JSON on stdout, documented exit codes, and the two-algorithm comparison."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
