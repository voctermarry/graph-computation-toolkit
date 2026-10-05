"""Storage backends: --storage selection, the stats storage report, and backend equivalence."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from graphtk.algorithms import bellman_ford, bfs, dfs, dijkstra
from graphtk.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main
from graphtk.errors import NegativeWeightError, ValidationError
from graphtk.graph import (
    DENSE_LIMIT_BYTES,
    Graph,
    csr_logical_bytes,
    dense_logical_bytes,
    select_storage,
)

EDGES = [
    {"source": "a", "target": "b", "weight": 1.0},
    {"source": "b", "target": "c", "weight": 2.0},
    {"source": "a", "target": "c", "weight": 5.0},
    {"source": "c", "target": "d", "weight": 1.0},
]

# A richer directed graph: zero-weight edge, self-loop, duplicate input row, negative weight.
RICH = [
    {"source": "a", "target": "b", "weight": 0},
    {"source": "b", "target": "c", "weight": -1},
    {"source": "a", "target": "c", "weight": 4},
    {"source": "c", "target": "c", "weight": 7},
    {"source": "c", "target": "d", "weight": 2},
    {"source": "c", "target": "d", "weight": 3},
]


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class StorageCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.edges = self.write_edges(EDGES)
        self.rich = self.write_edges(RICH, "rich.jsonl")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_edges(self, rows: list[dict], name: str = "edges.jsonl") -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    # -- output equivalence across backends ---------------------------------------------------------
    def test_every_command_is_byte_identical_across_backends(self) -> None:
        heuristic = os.path.join(self.directory.name, "heuristic.json")
        with open(heuristic, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"a": 3, "b": 2, "c": 1, "d": 0}))
        commands = [
            ["bfs", "--edges", self.edges, "--source", "a"],
            ["dijkstra", "--edges", self.edges, "--source", "a", "--target", "d"],
            ["dijkstra", "--edges", self.edges, "--directed", "--source", "a"],
            ["bellman-ford", "--edges", self.rich, "--directed", "--source", "a", "--target", "d"],
            ["astar", "--edges", self.edges, "--directed", "--source", "a", "--target", "d", "--heuristic", heuristic],
            ["components", "--edges", self.edges],
            ["components", "--edges", self.edges, "--directed"],
            ["toposort", "--edges", self.edges, "--directed"],
            ["pagerank", "--edges", self.edges, "--directed"],
            ["centrality", "--edges", self.edges],
            ["clustering", "--edges", self.edges],
            ["clustering", "--edges", self.rich, "--directed"],
            ["communities", "--edges", self.edges],
            ["compare", "--edges", self.edges, "--directed", "--source", "a"],
        ]
        for command in commands:
            with self.subTest(command=command[0], flags=command[2:]):
                baseline_code, baseline_out, baseline_err = run_cli(command)
                for storage in ("adjacency", "csr", "dense", "auto"):
                    code, out, err = run_cli([*command, "--storage", storage])
                    self.assertEqual((code, out, err), (baseline_code, baseline_out, baseline_err), storage)

    def test_error_documents_are_identical_across_backends(self) -> None:
        negative = self.write_edges([{"source": "a", "target": "b", "weight": -1}], "negative.jsonl")
        cycle = self.write_edges(
            [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}],
            "cycle.jsonl",
        )
        commands = [
            ["dijkstra", "--edges", negative, "--source", "a"],
            ["pagerank", "--edges", negative, "--directed"],
            ["toposort", "--edges", cycle, "--directed"],
            ["bfs", "--edges", self.edges, "--source", "missing"],
        ]
        for command in commands:
            with self.subTest(command=command[0]):
                baseline = run_cli(command)
                for storage in ("csr", "dense", "auto"):
                    self.assertEqual(run_cli([*command, "--storage", storage]), baseline, storage)

    # -- the stats storage report ---------------------------------------------------------------------
    def test_stats_without_storage_has_no_storage_field(self) -> None:
        code, out, _ = run_cli(["stats", "--edges", self.edges])
        self.assertEqual(code, EXIT_OK)
        self.assertNotIn("storage", json.loads(out))

    def test_stats_fields_outside_the_storage_object_are_unchanged(self) -> None:
        _, baseline, _ = run_cli(["stats", "--edges", self.edges])
        baseline_doc = json.loads(baseline)
        for storage in ("adjacency", "csr", "dense", "auto"):
            with self.subTest(storage=storage):
                code, out, _ = run_cli(["stats", "--edges", self.edges, "--storage", storage])
                self.assertEqual(code, EXIT_OK)
                document = json.loads(out)
                document.pop("storage")
                self.assertEqual(document, baseline_doc)

    def test_stats_reports_the_requested_and_selected_backend(self) -> None:
        # EDGES undirected: n = 4, a = 8 -> csr 8*5+16*8 = 168 bytes, dense 9*16 = 144 bytes.
        expected = {"csrBytes": 168, "denseBytes": 144, "adjacencyEntries": 8, "density": 0.5}
        cases = {
            "adjacency": ("adjacency", 168),
            "csr": ("csr", 168),
            "dense": ("dense", 144),
            "auto": ("dense", 144),  # 144 < 168: the smaller logical footprint wins
        }
        for requested, (selected, logical) in cases.items():
            with self.subTest(storage=requested):
                code, out, err = run_cli(["stats", "--edges", self.edges, "--storage", requested])
                self.assertEqual((code, err), (EXIT_OK, ""))
                document = json.loads(out)["storage"]
                self.assertEqual(document["requested"], requested)
                self.assertEqual(document["selected"], selected)
                self.assertEqual(document["logicalBytes"], logical)
                for key, value in expected.items():
                    self.assertEqual(document[key], value, key)

    def test_stats_storage_entries_count_directed_arcs_once(self) -> None:
        code, out, _ = run_cli(["stats", "--edges", self.edges, "--directed", "--storage", "auto"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)["storage"]
        self.assertEqual(document["adjacencyEntries"], 4)
        self.assertEqual(document["csrBytes"], 8 * 5 + 16 * 4)
        self.assertEqual(document["denseBytes"], 9 * 16)
        self.assertEqual(document["density"], 0.25)
        self.assertEqual(document["selected"], "csr")  # 104 < 144

    def test_stats_storage_counts_self_loops_once_and_undirected_edges_twice(self) -> None:
        rows = [
            {"source": "a", "target": "a"},  # self-loop: one entry
            {"source": "a", "target": "b"},  # undirected edge: two entries
        ]
        path = self.write_edges(rows, "loops.jsonl")
        code, out, _ = run_cli(["stats", "--edges", path, "--storage", "csr"])
        self.assertEqual(code, EXIT_OK)
        document = json.loads(out)["storage"]
        self.assertEqual(document["adjacencyEntries"], 3)
        self.assertEqual(document["density"], round(3 / 4, 10))

    def test_auto_on_an_empty_graph_selects_csr_and_reports_zero_density(self) -> None:
        empty = self.write_edges([], "empty.jsonl")
        code, out, _ = run_cli(["stats", "--edges", empty, "--storage", "auto"])
        self.assertEqual(code, EXIT_NEGATIVE)  # the empty-graph verdict is unchanged
        document = json.loads(out)["storage"]
        self.assertEqual(
            document,
            {
                "requested": "auto",
                "selected": "csr",
                "logicalBytes": 8,
                "csrBytes": 8,
                "denseBytes": 0,
                "adjacencyEntries": 0,
                "density": 0,
            },
        )

    def test_unknown_storage_is_a_validation_error_with_no_output(self) -> None:
        code, out, err = run_cli(["stats", "--edges", self.edges, "--storage", "triplestore"])
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertEqual(document["value"], "triplestore")

    def test_unknown_storage_wins_over_a_missing_file(self) -> None:
        missing = os.path.join(self.directory.name, "missing.jsonl")
        code, _, err = run_cli(["stats", "--edges", missing, "--storage", "nope"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(json.loads(err)["value"], "nope")

    # -- the dense limit --------------------------------------------------------------------------------
    def write_dense_busting_graph(self) -> str:
        # 7724 isolated-via-self-loop nodes: 9 * 7724^2 = 536941584 > 536870912.
        path = os.path.join(self.directory.name, "huge.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for index in range(7724):
                handle.write(json.dumps({"source": f"n{index}", "target": f"n{index}"}) + "\n")
        return path

    def test_explicit_dense_beyond_the_limit_is_refused_before_allocation(self) -> None:
        huge = self.write_dense_busting_graph()
        code, out, err = run_cli(["stats", "--edges", huge, "--storage", "dense"])
        self.assertEqual((code, out), (EXIT_ERROR, ""))
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertEqual(document["requestedBytes"], 9 * 7724 * 7724)
        self.assertEqual(document["limitBytes"], DENSE_LIMIT_BYTES)

    def test_auto_falls_back_to_csr_when_dense_would_exceed_the_limit(self) -> None:
        huge = self.write_dense_busting_graph()
        code, out, err = run_cli(["stats", "--edges", huge, "--storage", "auto"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)["storage"]
        self.assertEqual(document["selected"], "csr")
        self.assertEqual(document["denseBytes"], 9 * 7724 * 7724)
        self.assertEqual(document["logicalBytes"], document["csrBytes"])

    # -- zero-weight edges stay distinct from absent edges ----------------------------------------------
    def test_zero_weight_edge_is_visible_on_the_dense_backend(self) -> None:
        zero = self.write_edges([{"source": "a", "target": "b", "weight": 0}], "zero.jsonl")
        for storage in ("adjacency", "csr", "dense"):
            with self.subTest(storage=storage):
                code, out, _ = run_cli(["dijkstra", "--edges", zero, "--directed", "--source", "a", "--storage", storage])
                self.assertEqual(code, EXIT_OK)
                document = json.loads(out)
                self.assertEqual(document["distances"], {"a": 0.0, "b": 0.0})
                code, out, _ = run_cli(["stats", "--edges", zero, "--directed", "--storage", storage])
                self.assertEqual(json.loads(out)["edges"], 1)


class StorageSelectionTests(unittest.TestCase):
    """The deterministic auto formula and the accounting helpers."""

    def test_accounting_formulas(self) -> None:
        self.assertEqual(csr_logical_bytes(4, 8), 8 * 5 + 16 * 8)
        self.assertEqual(dense_logical_bytes(4), 9 * 16)
        self.assertEqual(csr_logical_bytes(0, 0), 8)
        self.assertEqual(dense_logical_bytes(0), 0)

    def test_auto_picks_the_smaller_and_csr_on_tie_or_empty(self) -> None:
        self.assertEqual(select_storage("auto", 4, 8), "dense")  # 144 < 168
        self.assertEqual(select_storage("auto", 4, 1), "csr")  # 56 < 144
        self.assertEqual(select_storage("auto", 0, 0), "csr")
        self.assertEqual(select_storage("auto", 7724, 7724), "csr")  # dense over the limit

    def test_explicit_dense_over_the_limit_raises_with_the_byte_evidence(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            select_storage("dense", 7724, 7724)
        self.assertEqual(caught.exception.context["requestedBytes"], 9 * 7724 * 7724)
        self.assertEqual(caught.exception.context["limitBytes"], DENSE_LIMIT_BYTES)

    def test_unknown_kind_raises(self) -> None:
        with self.assertRaises(ValidationError):
            select_storage("cylinder", 1, 1)
        with self.assertRaises(ValidationError):
            Graph(storage="cylinder")


class StorageBackendTests(unittest.TestCase):
    """The csr and dense backends genuinely serve the mutable Graph API."""

    def build(self, storage: str, directed: bool = True) -> Graph:
        graph = Graph(directed=directed, storage=storage)
        for row in RICH:
            graph.add_edge(row["source"], row["target"], row["weight"])
        return graph

    def test_backends_agree_with_the_adjacency_baseline(self) -> None:
        for directed in (True, False):
            with self.subTest(directed=directed):
                baseline = self.build("adjacency", directed)
                for storage in ("csr", "dense"):
                    graph = self.build(storage, directed)
                    self.assertEqual(graph.nodes(), baseline.nodes())
                    self.assertEqual(graph.edges(), baseline.edges())
                    self.assertEqual(graph.csr(), baseline.csr())
                    self.assertEqual(graph.adjacency_entry_count(), baseline.adjacency_entry_count())
                    for node in baseline.nodes():
                        self.assertEqual(graph.neighbors(node), baseline.neighbors(node))
                        self.assertEqual(graph.degree(node), baseline.degree(node))
                        self.assertEqual(graph.weighted_degree(node), baseline.weighted_degree(node))
                    self.assertEqual(graph.self_loops(), baseline.self_loops())
                    self.assertEqual(graph.to_document(), baseline.to_document())

    def test_algorithms_read_through_the_backends(self) -> None:
        baseline = self.build("adjacency")
        for storage in ("csr", "dense"):
            graph = self.build(storage)
            self.assertEqual(bfs(graph, "a"), bfs(baseline, "a"))
            self.assertEqual(dfs(graph, "a"), dfs(baseline, "a"))
            self.assertEqual(bellman_ford(graph, "a"), bellman_ford(baseline, "a"))
            # RICH has a negative edge: dijkstra must refuse with identical evidence on every backend.
            with self.assertRaises(NegativeWeightError) as raised:
                dijkstra(graph, "a")
            with self.assertRaises(NegativeWeightError) as raised_baseline:
                dijkstra(baseline, "a")
            self.assertEqual(raised.exception.to_document(), raised_baseline.exception.to_document())

    def test_dijkstra_agrees_on_a_non_negative_graph(self) -> None:
        rows = [row for row in RICH if row["weight"] >= 0]
        graphs = []
        for storage in ("adjacency", "csr", "dense"):
            graph = Graph(directed=True, storage=storage)
            for row in rows:
                graph.add_edge(row["source"], row["target"], row["weight"])
            graphs.append(graph)
        self.assertEqual(dijkstra(graphs[1], "a"), dijkstra(graphs[0], "a"))
        self.assertEqual(dijkstra(graphs[2], "a"), dijkstra(graphs[0], "a"))

    def test_mutation_is_immediately_visible_on_every_backend(self) -> None:
        for storage in ("adjacency", "csr", "dense"):
            with self.subTest(storage=storage):
                graph = Graph(directed=True, storage=storage)
                graph.add_node("lonely")
                graph.add_edge("a", "b", 1.5)
                graph.add_edge("a", "b", 2.5)  # duplicate overwrites
                graph.add_edge("b", "b", 0)  # zero-weight self-loop
                self.assertEqual(graph.nodes(), ["a", "b", "lonely"])
                self.assertEqual(graph.neighbors("a"), [("b", 2.5)])
                self.assertEqual(graph.neighbors("b"), [("b", 0.0)])
                self.assertEqual(graph.degree("b"), 1)
                self.assertEqual(graph.self_loops(), ["b"])
                self.assertEqual(bfs(graph, "a"), {"a": 0, "b": 1})
                self.assertTrue(graph.remove_edge("a", "b"))
                self.assertFalse(graph.remove_edge("a", "b"))
                self.assertEqual(graph.neighbors("a"), [])
                self.assertEqual(bfs(graph, "a"), {"a": 0})
                self.assertEqual(graph.adjacency_entry_count(), 1)

    def test_undirected_mirroring_and_removal_on_every_backend(self) -> None:
        for storage in ("adjacency", "csr", "dense"):
            with self.subTest(storage=storage):
                graph = Graph(storage=storage)
                graph.add_edge("a", "b", 3)
                self.assertEqual(graph.neighbors("b"), [("a", 3.0)])
                self.assertEqual(graph.adjacency_entry_count(), 2)
                self.assertTrue(graph.remove_edge("b", "a"))  # either direction removes both halves
                self.assertEqual(graph.neighbors("a"), [])
                self.assertEqual(graph.neighbors("b"), [])
                self.assertEqual(graph.edge_count(), 0)

    def test_to_storage_preserves_the_graph_including_isolated_nodes(self) -> None:
        graph = self.build("adjacency")
        graph.add_node("isolated")
        for storage in ("csr", "dense", "adjacency"):
            converted = graph.to_storage(storage)
            self.assertEqual(converted.storage, storage)
            self.assertEqual(converted.nodes(), graph.nodes())
            self.assertEqual(converted.edges(), graph.edges())
            self.assertEqual(converted.to_document(), graph.to_document())
            # And the converted graph stays mutable.
            converted.add_edge("isolated", "a", 1)
            self.assertEqual(dict(converted.neighbors("isolated")), {"a": 1.0})

    def test_from_edges_accepts_a_storage_kind(self) -> None:
        from graphtk.graph import Edge

        graph = Graph.from_edges([Edge("a", "b", 2)], storage="dense")
        self.assertEqual(graph.storage, "dense")
        self.assertEqual(graph.neighbors("b"), [("a", 2.0)])


if __name__ == "__main__":
    unittest.main()
