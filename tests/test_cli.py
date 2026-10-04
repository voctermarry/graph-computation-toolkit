"""CLI contract: JSON on stdout, documented exit codes, and the two-algorithm comparison."""

from __future__ import annotations

import contextlib
import io
import json
import os
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

    def test_clustering_reports_the_full_document_in_canonical_order(self) -> None:
        triangle = self.write_edges(
            [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
                {"source": "a", "target": "c"},
                {"source": "c", "target": "d"},
            ]
        )
        code, out, err = run_cli(["clustering", "--edges", triangle])
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertEqual(
            out,
            json.dumps(
                {
                    "average": 0.5833333333,
                    "coefficients": {"a": 1.0, "b": 1.0, "c": 0.3333333333, "d": 0.0},
                    "connectedTriples": 5,
                    "nodes": ["a", "b", "c", "d"],
                    "transitivity": 0.6,
                    "triangles": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )

    def test_clustering_matches_the_library_over_directed_mixed_orientation_edges(self) -> None:
        arcs = self.write_edges(
            [
                {"source": "a", "target": "b"},
                {"source": "c", "target": "b"},
                {"source": "a", "target": "c"},
                {"source": "b", "target": "a"},  # bidirectional pair must count once
                {"source": "a", "target": "a"},  # self-loop is neither neighbour nor triangle
            ]
        )
        code, out, _ = run_cli(["clustering", "--edges", arcs, "--directed"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)
        self.assertEqual(document["triangles"], 1)
        self.assertEqual(document["connectedTriples"], 3)
        self.assertEqual(document["transitivity"], 1.0)
        self.assertEqual(document["coefficients"], {"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertEqual(document["nodes"], ["a", "b", "c"])

    def test_clustering_is_byte_identical_across_insertion_orders(self) -> None:
        rows = [
            {"source": "a", "target": "b", "weight": 3.0},
            {"source": "b", "target": "c", "weight": -2.0},
            {"source": "a", "target": "c"},
            {"source": "c", "target": "d"},
            {"source": "d", "target": "b"},
        ]

        def emit(order: list[dict], directed: bool, tag: str) -> str:
            path = self.write_edges(order, f"order-{tag}.jsonl")
            argv = ["clustering", "--edges", path]
            if directed:
                argv.append("--directed")
            code, out, err = run_cli(argv)
            self.assertEqual((code, err), (EXIT_OK, ""))
            return out

        # Undirected: reversed order with swapped endpoints is the same simple graph.
        swapped = [
            {"source": row["target"], "target": row["source"], **({"weight": row["weight"]} if "weight" in row else {})}
            for row in reversed(rows)
        ]
        self.assertEqual(emit(rows, False, "u-a"), emit(swapped, False, "u-b"))
        # Directed: the same arcs in reverse insertion order.
        self.assertEqual(emit(rows, True, "d-a"), emit(list(reversed(rows)), True, "d-b"))

    def test_clustering_reads_the_edge_list_from_stdin(self) -> None:
        text = "".join(json.dumps(row) + "\n" for row in EDGES)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), unittest.mock.patch(
            "graphtk.cli.sys.stdin", io.StringIO(text)
        ):
            code = main(["clustering", "--edges", "-", "--directed"])
        self.assertEqual((code, err.getvalue()), (EXIT_OK, ""))
        document = json.loads(out.getvalue())
        self.assertEqual(document["nodes"], ["a", "b", "c", "d"])
        self.assertEqual(document["triangles"], 1)

    def test_clustering_empty_graph_is_a_complete_zero_document(self) -> None:
        empty = self.write_edges([])
        code, out, err = run_cli(["clustering", "--edges", empty])
        self.assertEqual((code, EXIT_NEGATIVE, err), (EXIT_NEGATIVE, EXIT_NEGATIVE, ""))
        self.assertEqual(
            json.loads(out),
            {
                "average": 0.0,
                "coefficients": {},
                "connectedTriples": 0,
                "nodes": [],
                "transitivity": 0.0,
                "triangles": 0,
            },
        )

    def test_clustering_parse_and_file_errors_exit_two(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
