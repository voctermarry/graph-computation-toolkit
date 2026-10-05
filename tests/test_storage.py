"""Storage backend contract: adjacency | csr | dense | auto.

The new backends are real read paths for neighbors/edges/degree and every algorithm, not a report
field: outputs must be byte-identical to the adjacency baseline across directed/undirected graphs,
self-loops, duplicate overwrites and zero-weight edges. These tests pin the footprint formula, the
auto decision (smaller logical footprint; CSR wins ties and every empty graph), the 512 MiB dense
refusal, mutation reflection, and the stats-only ``storage`` object.
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

from graphtk import (
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
    label_propagation,
    pagerank,
)
from graphtk.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main
from graphtk.graph import DENSE_LIMIT_BYTES

# Directed arcs, a reverse arc, a zero-weight arc, a self-loop and an isolated node: every special
# case the Graph semantics call out lives here at once. Weights stay non-negative so the algorithms
# that refuse negatives can run; negative-evidence parity has its own test below.
ROWS = [
    {"source": "a", "target": "b", "weight": 2.0},
    {"source": "a", "target": "c", "weight": 0.0},
    {"source": "b", "target": "c", "weight": 1.5},
    {"source": "c", "target": "b", "weight": 4.0},  # its own directed arc; reverse pair undirected
    {"source": "b", "target": "b", "weight": 3.0},  # self-loop
    {"source": "d", "target": "e", "weight": 1.0},
]
BACKENDS = ["adjacency", "csr", "dense", "auto"]

# sqrt(limit/9) lands between these: 9*7723^2 = 536,802,561 <= limit < 536,941,584 = 9*7724^2.
MAX_DENSE_N = 7723
MIN_TOO_LARGE_N = 7724
TOO_LARGE_BYTES = 9 * MIN_TOO_LARGE_N * MIN_TOO_LARGE_N


def build(directed: bool) -> Graph:
    graph = Graph(directed=directed)
    for row in ROWS:
        graph.add_edge(row["source"], row["target"], row["weight"])
    graph.add_node("lonely")
    return graph


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class FootprintFormulaTests(unittest.TestCase):
    def test_directed_entries_count_each_arc_once(self) -> None:
        graph = build(directed=True)
        # a:{b,c}=2, b:{c,b}=2, c:{b}=1, d:{e}=1, e:{}=0, lonely:{}=0 -> 6 adjacency entries.
        entries, csr_bytes, dense_bytes = graph.storage_sizes()
        self.assertEqual(entries, 6)
        n = 6
        self.assertEqual(csr_bytes, 8 * (n + 1) + 16 * entries)
        self.assertEqual(dense_bytes, 9 * n * n)

    def test_undirected_non_loop_counts_twice_and_self_loop_once(self) -> None:
        graph = build(directed=False)
        # Four distinct non-loop pairs (a-b, a-c, b-c -- written twice but one adjacency -- and
        # d-e), each reachable from both ends = 4*2, plus one self-loop (b) = 9.
        entries, csr_bytes, dense_bytes = graph.storage_sizes()
        self.assertEqual(entries, 9)
        n = 6
        self.assertEqual(csr_bytes, 8 * (n + 1) + 16 * entries)
        self.assertEqual(dense_bytes, 9 * n * n)

    def test_empty_graph_reports_zero_entries_and_offset_array_only(self) -> None:
        graph = Graph()
        entries, csr_bytes, dense_bytes = graph.storage_sizes()
        self.assertEqual((entries, dense_bytes), (0, 0))
        self.assertEqual(csr_bytes, 8)  # 8*(0+1) + 0


class AutoSelectionTests(unittest.TestCase):
    def test_empty_graph_auto_selects_csr_even_though_zero_dense_bytes_would_be_smaller(self) -> None:
        document = Graph().configure_storage("auto")
        self.assertEqual(document["selected"], "csr")
        self.assertEqual(document["requested"], "auto")

    def test_auto_picks_csr_for_a_sparse_graph(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b")
        graph.add_edge("c", "d")
        document = graph.configure_storage("auto")
        # n=4, a=4: csr = 40+64 = 104, dense = 144 -> csr.
        self.assertEqual(document["selected"], "csr")
        self.assertEqual(document["csrBytes"], 104)
        self.assertEqual(document["denseBytes"], 144)

    def test_auto_picks_dense_when_it_is_strictly_smaller(self) -> None:
        graph = Graph()
        for left in ("a", "b", "c"):
            for right in ("a", "b", "c"):
                graph.add_edge(left, right)
        document = graph.configure_storage("auto")
        # a = 9 (every cell filled once per row): csr = 32+144 = 176, dense = 81 -> dense.
        self.assertEqual(document["selected"], "dense")
        self.assertEqual(document["adjacencyEntries"], 9)
        self.assertEqual(document["logicalBytes"], 81)

    def test_auto_decision_always_follows_the_formula(self) -> None:
        # Independently re-derive the rule for a family of graphs and check the reported selection.
        for directed in (False, True):
            graph = Graph(directed=directed)
            for index in range(20):
                graph.add_node(f"n{index}")
                if index:
                    graph.add_edge("n0", f"n{index}")
                if index > 1:
                    graph.add_edge(f"n{index - 1}", f"n{index}")
                if not directed and index > 2:
                    graph.add_edge("n1", f"n{index}")
                if index % 4 == 0:
                    graph.add_edge(f"n{index}", f"n{index}")
                entries, csr_bytes, dense_bytes = graph.storage_sizes()
                n = graph.node_count
                expected = "csr" if n == 0 or csr_bytes <= dense_bytes else "dense"
                document = graph.configure_storage("auto")
                self.assertEqual(document["selected"], expected, (directed, index))

    def test_auto_oversized_dense_falls_back_to_csr_without_error(self) -> None:
        graph = Graph(directed=True)
        for index in range(MIN_TOO_LARGE_N):
            graph.add_node(f"n{index}")
        document = graph.configure_storage("auto")
        self.assertEqual(document["selected"], "csr")
        self.assertGreater(document["denseBytes"], DENSE_LIMIT_BYTES)
        self.assertEqual(len(graph.nodes()), MIN_TOO_LARGE_N)


class DenseLimitTests(unittest.TestCase):
    def test_explicit_dense_over_the_limit_is_refused_before_allocation(self) -> None:
        graph = Graph(directed=True)
        for index in range(MIN_TOO_LARGE_N):
            graph.add_node(f"n{index}")
        with self.assertRaises(ValidationError) as caught:
            graph.configure_storage("dense")
        context = caught.exception.context
        self.assertEqual(context["requestedBytes"], TOO_LARGE_BYTES)
        self.assertEqual(context["limitBytes"], DENSE_LIMIT_BYTES)
        self.assertGreater(context["requestedBytes"], context["limitBytes"])

    def test_dense_at_or_under_the_limit_is_built(self) -> None:
        graph = Graph(directed=True)
        for index in range(MAX_DENSE_N):
            graph.add_node(f"n{index}")
        document = graph.configure_storage("dense")
        self.assertEqual(document["selected"], "dense")
        self.assertLessEqual(document["denseBytes"], DENSE_LIMIT_BYTES)

    def test_cli_dense_over_limit_is_code_two_json_with_bytes_and_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "big.jsonl")
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                for index in range(MIN_TOO_LARGE_N):
                    handle.write(f'{{"source":"n{index}","target":"n{index}"}}\n')
            code, out, err = run_cli(["stats", "--edges", path, "--directed", "--storage", "dense"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertEqual(document["requestedBytes"], TOO_LARGE_BYTES)
        self.assertEqual(document["limitBytes"], DENSE_LIMIT_BYTES)
        self.assertEqual(err.count("\n"), 1)

    def test_cli_unknown_storage_is_code_two_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "edges.jsonl")
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write('{"source":"a","target":"b"}\n')
            for storage in ("matrix", "CSR", "denser"):
                with self.subTest(storage=storage):
                    code, out, err = run_cli(["stats", "--edges", path, "--storage", storage])
                    self.assertEqual(code, EXIT_ERROR)
                    self.assertEqual(out, "")
                    document = json.loads(err)
                    self.assertEqual(document["error"], "validation_error")

    def test_unknown_storage_is_rejected_before_the_edge_file_is_read(self) -> None:
        # A missing file must not mask the unknown-storage validation.
        code, out, err = run_cli(["stats", "--edges", "/no/such/file.jsonl", "--storage", "bogus"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        self.assertEqual(json.loads(err)["error"], "validation_error")


class BackendParityTests(unittest.TestCase):
    """Reads and algorithms must agree exactly across all four selections."""

    def reference(self, graph: Graph) -> Graph:
        baseline = Graph(directed=graph.directed)
        for node in graph.nodes():
            baseline.add_node(node)
        for edge in graph.edges():
            baseline.add_edge(edge.source, edge.target, edge.weight)
        return baseline

    def assert_same_reads(self, graph: Graph) -> None:
        baseline = self.reference(graph)
        self.assertEqual(graph.nodes(), baseline.nodes())
        self.assertEqual(graph.edges(), baseline.edges())
        self.assertEqual(graph.self_loops(), baseline.self_loops())
        self.assertEqual(graph.to_document(), baseline.to_document())
        for node in graph.nodes():
            self.assertEqual(graph.neighbors(node), baseline.neighbors(node), node)
            self.assertEqual(graph.degree(node), baseline.degree(node), node)
        nodes, offsets, flat = graph.csr()
        base_nodes, base_offsets, base_flat = baseline.csr()
        self.assertEqual((nodes, offsets, flat), (base_nodes, base_offsets, base_flat))

    def test_read_paths_agree_for_every_backend(self) -> None:
        for directed in (False, True):
            for backend in BACKENDS:
                with self.subTest(directed=directed, backend=backend):
                    graph = build(directed)
                    graph.configure_storage(backend)
                    self.assert_same_reads(graph)

    def test_zero_weight_edge_is_distinct_from_no_edge(self) -> None:
        for backend in ("csr", "dense", "auto"):
            with self.subTest(backend=backend):
                graph = Graph(directed=True)
                graph.add_edge("a", "b", 0.0)
                graph.add_node("c")
                graph.configure_storage(backend)
                # Present with weight 0...
                self.assertEqual(graph.neighbors("a"), [("b", 0.0)])
                self.assertEqual(graph.degree("a"), 1)
                # ...an absent pair returns nothing, an unknown node still raises.
                self.assertEqual(graph.neighbors("b"), [])
                self.assertEqual(graph.neighbors("c"), [])
                with self.assertRaises(ValidationError):
                    graph.neighbors("zzz")
                edge = graph.edges()[0]
                self.assertEqual((edge.source, edge.target, edge.weight), ("a", "b", 0.0))

    def test_algorithms_agree_across_backends(self) -> None:
        for directed in (False, True):
            reference = build(directed)
            expected = {
                "nodes": reference.nodes(),
                "edges": [(e.source, e.target, e.weight) for e in reference.edges()],
                "bfs": bfs(reference, "a"),
                "dfs": dfs(reference, "a"),
                "dijkstra": dijkstra(reference, "a"),
                "bellman": bellman_ford(reference, "a"),
                "components": components(reference),
                "centrality": degree_centrality(reference),
                "clustering": clustering(reference).to_document(),
                "pagerank": pagerank(reference).to_document(),
                "communities": label_propagation(reference).to_document(),
            }
            for backend in ("csr", "dense", "auto"):
                with self.subTest(directed=directed, backend=backend):
                    graph = build(directed)
                    graph.configure_storage(backend)
                    self.assertEqual(graph.nodes(), expected["nodes"])
                    self.assertEqual(
                        [(e.source, e.target, e.weight) for e in graph.edges()], expected["edges"]
                    )
                    self.assertEqual(bfs(graph, "a"), expected["bfs"])
                    self.assertEqual(dfs(graph, "a"), expected["dfs"])
                    self.assertEqual(dijkstra(graph, "a"), expected["dijkstra"])
                    self.assertEqual(bellman_ford(graph, "a"), expected["bellman"])
                    self.assertEqual(components(graph), expected["components"])
                    self.assertEqual(degree_centrality(graph), expected["centrality"])
                    self.assertEqual(clustering(graph).to_document(), expected["clustering"])
                    self.assertEqual(pagerank(graph).to_document(), expected["pagerank"])
                    self.assertEqual(label_propagation(graph).to_document(), expected["communities"])

    def test_self_loops_and_overwrites_survive_the_switch(self) -> None:
        graph = Graph()
        graph.add_edge("a", "a", 2.0)
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("a", "b", 5.0)  # last write wins on both halves
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                graph.configure_storage(backend)
                self.assertEqual(graph.self_loops(), ["a"])
                self.assertEqual(graph.neighbors("a"), [("a", 2.0), ("b", 5.0)])
                self.assertEqual(graph.neighbors("b"), [("a", 5.0)])
                self.assertEqual(graph.edge_count(), 2)

    def test_negative_weight_and_cycle_evidence_is_identical_on_every_backend(self) -> None:
        def negative_graph() -> Graph:
            graph = Graph(directed=True)
            graph.add_edge("a", "b", 1.0)
            graph.add_edge("b", "c", -2.0)
            graph.add_edge("c", "a", -1.0)  # negative cycle too
            return graph

        def evidence(document: dict) -> dict:
            # The message names the algorithm, but the evidence (edge/weight) must not move.
            return {key: value for key, value in document.items() if key != "message"}

        for backend in ("adjacency", "csr", "dense", "auto"):
            graph = negative_graph()
            graph.configure_storage(backend)
            with self.assertRaises(NegativeWeightError) as caught:
                dijkstra(graph, "a")
            dijkstra_evidence = evidence(caught.exception.to_document())
            with self.assertRaises(NegativeWeightError) as caught:
                pagerank(graph)
            self.assertEqual(evidence(caught.exception.to_document()), dijkstra_evidence)
            with self.assertRaises(NegativeWeightError) as caught:
                label_propagation(graph)
            self.assertEqual(evidence(caught.exception.to_document()), dijkstra_evidence)
            with self.assertRaises(NegativeCycleError) as caught:
                bellman_ford(graph, "a")
            self.assertEqual(caught.exception.to_document()["error"], "negative_cycle_error")
            self.assertEqual(dijkstra_evidence, {"error": "negative_weight_error", "edge": "b->c", "weight": -2.0})


class MutationReflectionTests(unittest.TestCase):
    def test_mutations_after_switch_are_visible_immediately(self) -> None:
        for backend in ("csr", "dense"):
            with self.subTest(backend=backend):
                graph = build(directed=True)
                graph.configure_storage(backend)
                graph.add_node("fresh")
                graph.add_edge("a", "fresh", 7.0)
                graph.add_edge("a", "b", 9.0)  # overwrite
                self.assertEqual(graph.neighbors("a"), [("b", 9.0), ("c", 0.0), ("fresh", 7.0)])
                self.assertIn("fresh", graph.nodes())
                self.assertTrue(graph.remove_edge("d", "e"))
                self.assertEqual(graph.neighbors("d"), [])
                self.assertEqual(graph.edge_count(), 6)  # a-b, a-c, b-c, c-b, b-b, a-fresh
                self.assertEqual(bfs(graph, "d"), {"d": 0})

    def test_switching_backends_keeps_the_mutable_api(self) -> None:
        graph = build(directed=False)
        graph.configure_storage("dense")
        graph.configure_storage("csr")
        graph.add_edge("lonely", "a")
        self.assertEqual(graph.neighbors("lonely"), [("a", 1.0)])
        graph.configure_storage("adjacency")
        self.assertEqual(graph.neighbors("lonely"), [("a", 1.0)])
        self.assertTrue(graph.remove_edge("lonely", "a"))

    def test_dense_rebuild_after_growth_past_the_limit_is_refused(self) -> None:
        graph = Graph(directed=True)
        for index in range(2):
            graph.add_node(f"n{index}")
        graph.configure_storage("dense")
        self.assertEqual(graph.neighbors("n0"), [])
        for index in range(2, MIN_TOO_LARGE_N):
            graph.add_node(f"n{index}")
        with self.assertRaises(ValidationError) as caught:
            graph.nodes()
        self.assertEqual(caught.exception.context["requestedBytes"], TOO_LARGE_BYTES)
        self.assertEqual(caught.exception.context["limitBytes"], DENSE_LIMIT_BYTES)


class StatsStorageDocumentTests(unittest.TestCase):
    def write(self, directory: str, rows: list[dict], name: str = "edges.jsonl") -> str:
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def test_stats_without_storage_has_no_storage_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, [{"source": "a", "target": "b"}])
            code, out, err = run_cli(["stats", "--edges", path])
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertNotIn("storage", json.loads(out))

    def test_stats_storage_document_is_stable_for_every_selection(self) -> None:
        rows = [{"source": "a", "target": "b", "weight": 0}, {"source": "b", "target": "c"}]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, rows)
            documents = {}
            for backend in BACKENDS:
                code, out, err = run_cli(["stats", "--edges", path, "--storage", backend])
                self.assertEqual((code, err), (EXIT_OK, ""))
                documents[backend] = json.loads(out)["storage"]
        # n=3, undirected: a=4; csr = 32+64 = 96, dense = 81.
        for document in documents.values():
            self.assertEqual(document["adjacencyEntries"], 4)
            self.assertEqual(document["csrBytes"], 96)
            self.assertEqual(document["denseBytes"], 81)
            self.assertEqual(document["density"], round(4 / 9, 10))
        self.assertEqual(documents["adjacency"]["requested"], "adjacency")
        self.assertEqual(documents["adjacency"]["selected"], "adjacency")
        self.assertEqual(documents["adjacency"]["logicalBytes"], 96)
        self.assertEqual(documents["csr"]["requested"], "csr")
        self.assertEqual(documents["csr"]["selected"], "csr")
        self.assertEqual(documents["csr"]["logicalBytes"], 96)
        self.assertEqual(documents["dense"]["requested"], "dense")
        self.assertEqual(documents["dense"]["selected"], "dense")
        self.assertEqual(documents["dense"]["logicalBytes"], 81)
        self.assertEqual(documents["auto"]["requested"], "auto")
        self.assertEqual(documents["auto"]["selected"], "dense")
        self.assertEqual(documents["auto"]["logicalBytes"], 81)

    def test_stats_empty_graph_density_is_zero_and_auto_selects_csr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "empty.jsonl")
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write("# nothing\n\n")
            code, out, err = run_cli(["stats", "--edges", path, "--storage", "auto"])
        self.assertEqual(code, EXIT_NEGATIVE)
        self.assertEqual(err, "")
        storage = json.loads(out)["storage"]
        self.assertEqual(storage["selected"], "csr")
        self.assertEqual((storage["adjacencyEntries"], storage["density"]), (0, 0))
        self.assertEqual((storage["csrBytes"], storage["denseBytes"]), (8, 0))

    def test_other_commands_gain_no_fields_and_match_byte_for_byte(self) -> None:
        rows = [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "c"},
            {"source": "a", "target": "c"},
        ]
        commands = [
            ["bfs", "--source", "a"],
            ["dijkstra", "--source", "a"],
            ["bellman-ford", "--source", "a"],
            ["components"],
            ["centrality"],
            ["clustering"],
            ["compare", "--source", "a"],
            ["pagerank"],
            ["communities"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, rows)
            for command in commands:
                outputs = {}
                for backend in ("adjacency", "csr", "dense", "auto"):
                    flag = [] if backend == "adjacency" else ["--storage", backend]
                    code, out, err = run_cli([*command, "--edges", path, *flag])
                    self.assertEqual((code, err), (EXIT_OK, ""), (command, backend))
                    document = json.loads(out)
                    self.assertNotIn("storage", document)
                    outputs[backend] = document
                self.assertEqual(set(outputs["csr"]), set(outputs["adjacency"]), command)
                self.assertEqual(outputs["csr"], outputs["adjacency"], command)
                self.assertEqual(outputs["dense"], outputs["adjacency"], command)
                self.assertEqual(outputs["auto"], outputs["adjacency"], command)

    def test_directed_commands_match_byte_for_byte_on_every_backend(self) -> None:
        rows = [
            {"source": "a", "target": "b", "weight": 2},
            {"source": "b", "target": "c", "weight": 1},
            {"source": "a", "target": "c", "weight": 5},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, rows)
            heuristic = os.path.join(directory, "h.json")
            with open(heuristic, "w", encoding="utf-8") as handle:
                json.dump({"a": 3.0, "b": 1.0, "c": 0.0}, handle)
            cases = [
                ["toposort"],
                ["astar", "--source", "a", "--target", "c", "--heuristic", heuristic],
                ["dijkstra", "--source", "a", "--target", "c"],
            ]
            for command in cases:
                outputs = {}
                for backend in ("csr", "dense", "auto"):
                    code, out, err = run_cli(
                        [*command, "--edges", path, "--directed", "--storage", backend]
                    )
                    self.assertEqual((code, err), (EXIT_OK, ""), (command, backend))
                    outputs[backend] = out
                code, base, err = run_cli([*command, "--edges", path, "--directed"])
                self.assertEqual((code, err), (EXIT_OK, ""), command)
                self.assertEqual(outputs["csr"], base, command)
                self.assertEqual(outputs["dense"], base, command)
                self.assertEqual(outputs["auto"], base, command)

    def test_storage_flag_reads_stdin(self) -> None:
        text = '{"source":"a","target":"b"}\n'
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), unittest.mock.patch.object(
            sys, "stdin", io.StringIO(text)
        ):
            code = main(["stats", "--edges", "-", "--storage", "csr"])
        self.assertEqual((code, err.getvalue()), (EXIT_OK, ""))
        self.assertEqual(json.loads(out.getvalue())["storage"]["selected"], "csr")


if __name__ == "__main__":
    unittest.main()
