"""Shortest-path consistency regression: three algorithms, two entry points, one answer.

Every graph in the battery uses finite integer weights so an exact float comparison can never
mistake rounding noise for an algorithmic disagreement. For each non-negative graph and each
source, Dijkstra, Bellman-Ford and A* (with an all-zero heuristic) must report the same reachable
set and the same distances -- through both the Python API and the CLI. Where several shortest
routes tie, the algorithms may pick different ones, but every returned route must start at the
source, end at the target, walk real traversable edges, and sum to the reported distance.

Negative weights are only exercised through Bellman-Ford, checked against an independent
simple-path oracle; Dijkstra, A* and compare must keep refusing them with NegativeWeightError.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
import unittest.mock

from graphtk import (
    Graph,
    NegativeCycleError,
    NegativeWeightError,
    ValidationError,
    astar,
    bellman_ford,
    dijkstra,
    path_from,
)
from graphtk.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


# -- the graph battery ----------------------------------------------------------------------------
# One definition per graph -- name, directedness, JSONL-ready rows -- so the Python-level and the
# CLI-level tests exercise exactly the same graph, duplicate rows and all. Weights stay integral.
BATTERY: list[tuple[str, bool, list[tuple[str, str, int]]]] = [
    # Undirected, connected, with a tie: a->c costs 4 directly and 2 + 2 via b.
    (
        "undirected_connected",
        False,
        [("a", "b", 2), ("b", "c", 2), ("a", "c", 4), ("c", "d", 3), ("d", "e", 1), ("c", "e", 6)],
    ),
    # Directed and connected, with equal-length candidates: a reaches c for 2 both directly and
    # via b, and reaches d for 3 via b and via c.
    (
        "directed_connected",
        True,
        [("a", "b", 1), ("a", "c", 2), ("b", "c", 1), ("b", "d", 2), ("c", "d", 1), ("d", "e", 4), ("a", "e", 9)],
    ),
    # Two components plus a node only a self-loop touches: from any source, some nodes are
    # unreachable and must simply be absent from the distance map.
    (
        "disconnected_directed",
        True,
        [("a", "b", 3), ("c", "d", 2), ("e", "e", 4)],
    ),
    # Zero-weight edges: the cheapest route costs nothing at all for two hops.
    (
        "zero_weight_edges",
        True,
        [("a", "b", 0), ("b", "c", 0), ("a", "c", 5), ("c", "d", 1)],
    ),
    # Self-loops, one of them zero-weight: never part of a shortest route, never an error.
    (
        "self_loops",
        False,
        [("a", "a", 7), ("a", "b", 1), ("b", "b", 0), ("b", "c", 2)],
    ),
    # A repeated edge: the later row overwrites the earlier one, so a->b costs 2, not 9.
    (
        "duplicate_edges",
        True,
        [("a", "b", 9), ("a", "b", 2), ("b", "c", 1)],
    ),
    # From source a only a itself is reachable; the b<->c pair is a separate component.
    (
        "single_reachable_node",
        True,
        [("a", "a", 5), ("b", "c", 1), ("c", "b", 1)],
    ),
    # A square of unit edges: two equal shortest routes from a to d.
    (
        "equal_length_routes",
        False,
        [("a", "b", 1), ("b", "d", 1), ("a", "c", 1), ("c", "d", 1)],
    ),
]


def build_graph(directed: bool, rows: list[tuple[str, str, int]]) -> Graph:
    graph = Graph(directed=directed)
    for source, target, weight in rows:
        graph.add_edge(source, target, weight)
    return graph


def traversable_weights(graph: Graph) -> dict[tuple[str, str], float]:
    """Every directed step the graph actually allows, with its weight."""
    return {(node, neighbour): weight for node in graph.nodes() for neighbour, weight in graph.neighbors(node)}


def assert_real_path(
    case: unittest.TestCase,
    weights: dict[tuple[str, str], float],
    path: list[str],
    source: str,
    target: str,
    distance: float,
) -> None:
    """A returned route must walk real edges from source to target and sum to the distance."""
    case.assertEqual(path[0], source)
    case.assertEqual(path[-1], target)
    total = 0.0
    for step_source, step_target in zip(path, path[1:]):
        case.assertIn((step_source, step_target), weights)
        total += weights[(step_source, step_target)]
    case.assertEqual(total, distance)


def simple_path_distances(graph: Graph, source: str) -> dict[str, float]:
    """Independent oracle: the cheapest simple path to every node, by exhaustive enumeration.

    With no reachable negative cycle a shortest walk never repeats a node, so the minimum over
    simple paths is the true shortest distance -- computed here without any relaxation machinery.
    """
    best = {source: 0.0}

    def walk(node: str, cost: float, seen: frozenset[str]) -> None:
        for neighbour, weight in graph.neighbors(node):
            if neighbour in seen:
                continue
            candidate = cost + weight
            if neighbour not in best or candidate < best[neighbour]:
                best[neighbour] = candidate
            walk(neighbour, candidate, seen | {neighbour})

    walk(source, 0.0, frozenset({source}))
    return best


# -- Python entry point ---------------------------------------------------------------------------
class CrossAlgorithmAgreementTests(unittest.TestCase):
    def test_distances_agree_across_all_three_algorithms(self) -> None:
        for name, directed, rows in BATTERY:
            graph = build_graph(directed, rows)
            for source in graph.nodes():
                with self.subTest(graph=name, source=source):
                    dijkstra_distance, _ = dijkstra(graph, source)
                    bellman_distance, _ = bellman_ford(graph, source)
                    self.assertEqual(sorted(dijkstra_distance), sorted(bellman_distance))
                    for node in dijkstra_distance:
                        self.assertEqual(dijkstra_distance[node], bellman_distance[node])
                    zero_heuristic = {node: 0 for node in graph.nodes()}
                    for target in dijkstra_distance:
                        distance, _, _ = astar(graph, source, target, zero_heuristic)
                        self.assertEqual(distance, dijkstra_distance[target])

    def test_every_returned_path_is_real_and_sums_to_the_distance(self) -> None:
        for name, directed, rows in BATTERY:
            graph = build_graph(directed, rows)
            weights = traversable_weights(graph)
            zero_heuristic = {node: 0 for node in graph.nodes()}
            for source in graph.nodes():
                dijkstra_distance, dijkstra_previous = dijkstra(graph, source)
                bellman_distance, bellman_previous = bellman_ford(graph, source)
                for target in dijkstra_distance:
                    with self.subTest(graph=name, source=source, target=target):
                        _, astar_previous, _ = astar(graph, source, target, zero_heuristic)
                        for previous in (dijkstra_previous, bellman_previous, astar_previous):
                            path = path_from(previous, target)
                            assert_real_path(self, weights, path, source, target, dijkstra_distance[target])
                        self.assertEqual(bellman_distance[target], dijkstra_distance[target])

    def test_duplicate_rows_keep_the_last_weight(self) -> None:
        graph = build_graph(True, [("a", "b", 9), ("a", "b", 2), ("b", "c", 1)])
        distance, _ = dijkstra(graph, "a")
        self.assertEqual(distance, {"a": 0.0, "b": 2.0, "c": 3.0})

    def test_single_reachable_node_reports_exactly_the_source(self) -> None:
        graph = build_graph(True, [("a", "a", 5), ("b", "c", 1), ("c", "b", 1)])
        for distances in (dijkstra(graph, "a")[0], bellman_ford(graph, "a")[0]):
            self.assertEqual(distances, {"a": 0.0})


class NegativeWeightTests(unittest.TestCase):
    """Bellman-Ford alone handles negative edges; the others must keep refusing them."""

    def graph(self) -> Graph:
        # No negative cycle: from a, the distances are b=1, c=-1 (via b), d=0 (via b, c).
        return build_graph(
            True,
            [("a", "b", 1), ("a", "c", 4), ("b", "c", -2), ("c", "d", 1), ("b", "d", 5), ("e", "f", 3)],
        )

    def test_bellman_ford_matches_the_independent_oracle(self) -> None:
        graph = self.graph()
        distance, _ = bellman_ford(graph, "a")
        self.assertEqual(distance, simple_path_distances(graph, "a"))
        self.assertEqual(distance, {"a": 0.0, "b": 1.0, "c": -1.0, "d": 0.0})

    def test_reconstructed_paths_are_real_and_sum_to_the_distance(self) -> None:
        graph = self.graph()
        weights = traversable_weights(graph)
        distance, previous = bellman_ford(graph, "a")
        for target in distance:
            with self.subTest(target=target):
                assert_real_path(self, weights, path_from(previous, target), "a", target, distance[target])

    def test_dijkstra_and_astar_keep_refusing_negative_weights(self) -> None:
        graph = self.graph()
        with self.assertRaises(NegativeWeightError):
            dijkstra(graph, "a")
        with self.assertRaises(NegativeWeightError):
            astar(graph, "a", "d", {node: 0 for node in graph.nodes()})


class NegativeCycleTests(unittest.TestCase):
    def graph(self) -> Graph:
        # b -> c -> b totals -1, reachable from a.
        return build_graph(True, [("a", "b", 1), ("b", "c", -2), ("c", "b", 1)])

    def test_bellman_ford_raises_a_stable_negative_cycle_error(self) -> None:
        graph = self.graph()
        with self.assertRaises(NegativeCycleError) as first:
            bellman_ford(graph, "a")
        with self.assertRaises(NegativeCycleError) as second:
            bellman_ford(graph, "a")
        self.assertEqual(first.exception.to_document(), second.exception.to_document())


class ValidationFailureTests(unittest.TestCase):
    def test_unknown_source_is_a_validation_error_for_every_algorithm(self) -> None:
        graph = build_graph(False, [("a", "b", 1)])
        with self.assertRaises(ValidationError):
            dijkstra(graph, "missing")
        with self.assertRaises(ValidationError):
            bellman_ford(graph, "missing")
        with self.assertRaises(ValidationError):
            astar(graph, "missing", "a", None)

    def test_path_from_an_unreachable_target_is_a_validation_error(self) -> None:
        graph = build_graph(True, [("a", "b", 3), ("c", "d", 2)])
        _, previous = dijkstra(graph, "a")
        with self.assertRaises(ValidationError):
            path_from(previous, "c")

    def test_astar_with_an_unreachable_target_is_a_validation_error(self) -> None:
        graph = build_graph(True, [("a", "b", 3), ("c", "d", 2)])
        with self.assertRaises(ValidationError):
            astar(graph, "a", "c", {node: 0 for node in graph.nodes()})


# -- CLI entry point ------------------------------------------------------------------------------
class CompareCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_edges(self, rows: list[tuple[str, str, int]], name: str = "edges.jsonl") -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for source, target, weight in rows:
                handle.write(json.dumps({"source": source, "target": target, "weight": weight}) + "\n")
        return path

    def write_heuristic(self, graph: Graph, name: str = "heuristic.json") -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({node: 0 for node in graph.nodes()}))
        return path

    def assert_error_document(self, out: str, err: str, kind: str) -> dict:
        """A failure: nothing on stdout, exactly one established-format JSON document on stderr."""
        self.assertEqual(out, "")
        lines = err.strip().splitlines()
        self.assertEqual(len(lines), 1)
        document = json.loads(lines[0])
        self.assertEqual(document["error"], kind)
        self.assertIn("message", document)
        return document

    def test_cli_distances_and_paths_agree_for_every_battery_graph(self) -> None:
        for name, directed, rows in BATTERY:
            graph = build_graph(directed, rows)
            weights = traversable_weights(graph)
            edges = self.write_edges(rows, f"{name}.jsonl")
            heuristic = self.write_heuristic(graph, f"{name}.heuristic.json")
            flag = ["--directed"] if directed else []
            for source in graph.nodes():
                with self.subTest(graph=name, source=source):
                    api_distance, _ = dijkstra(graph, source)
                    expected_code = EXIT_OK if len(api_distance) > 1 else EXIT_NEGATIVE

                    d_code, d_out, d_err = run_cli(["dijkstra", "--edges", edges, *flag, "--source", source])
                    b_code, b_out, b_err = run_cli(["bellman-ford", "--edges", edges, *flag, "--source", source])
                    self.assertEqual((d_code, d_err), (expected_code, ""))
                    self.assertEqual((b_code, b_err), (expected_code, ""))
                    dijkstra_doc = json.loads(d_out)
                    bellman_doc = json.loads(b_out)
                    # The CLI document is the API answer, rounded and with unreachable nodes omitted.
                    self.assertEqual(dijkstra_doc["distances"], {node: round(value, 10) for node, value in sorted(api_distance.items())})
                    self.assertEqual(dijkstra_doc["distances"], bellman_doc["distances"])
                    self.assertEqual(dijkstra_doc["reachable"], bellman_doc["reachable"])

                    for target in api_distance:
                        a_code, a_out, a_err = run_cli(
                            ["astar", "--edges", edges, *flag, "--source", source, "--target", target, "--heuristic", heuristic]
                        )
                        self.assertEqual((a_code, a_err), (EXIT_OK, ""))
                        astar_doc = json.loads(a_out)
                        self.assertEqual(astar_doc["distance"], round(api_distance[target], 10))
                        assert_real_path(self, weights, astar_doc["path"], source, target, api_distance[target])

                        t_code, t_out, t_err = run_cli(
                            ["dijkstra", "--edges", edges, *flag, "--source", source, "--target", target]
                        )
                        self.assertEqual((t_code, t_err), (EXIT_OK, ""))
                        target_doc = json.loads(t_out)
                        self.assertEqual(target_doc["distance"], round(api_distance[target], 10))
                        assert_real_path(self, weights, target_doc["path"], source, target, api_distance[target])

    def test_compare_reports_identical_with_reachable_and_compared_counts(self) -> None:
        rows = [("a", "b", 1), ("a", "c", 2), ("b", "c", 1), ("b", "d", 2), ("c", "d", 1), ("d", "e", 4), ("a", "e", 9)]
        edges = self.write_edges(rows)
        code, out, err = run_cli(["compare", "--edges", edges, "--directed", "--source", "a"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertTrue(document["identical"])
        self.assertEqual(document["nodesCompared"], 5)
        self.assertEqual(document["dijkstraReachable"], 5)
        self.assertEqual(document["bellmanFordReachable"], 5)
        self.assertNotIn("differences", document)

    def test_compare_counts_only_the_reachable_nodes(self) -> None:
        edges = self.write_edges([("a", "b", 3), ("c", "d", 2)])
        code, out, err = run_cli(["compare", "--edges", edges, "--directed", "--source", "a"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertTrue(document["identical"])
        self.assertEqual(document["nodesCompared"], 2)
        self.assertEqual(document["dijkstraReachable"], 2)
        self.assertEqual(document["bellmanFordReachable"], 2)

    def test_compare_stdout_is_byte_identical_when_the_input_order_changes(self) -> None:
        rows = [("a", "b", 2), ("b", "c", 2), ("a", "c", 4), ("a", "b", 1), ("c", "c", 3), ("c", "d", 1)]
        first = self.write_edges(rows, "first.jsonl")
        second = self.write_edges(list(reversed(rows)), "second.jsonl")
        code_a, out_a, err_a = run_cli(["compare", "--edges", first, "--directed", "--source", "a"])
        code_b, out_b, err_b = run_cli(["compare", "--edges", second, "--directed", "--source", "a"])
        self.assertEqual((code_a, err_a), (EXIT_OK, ""))
        self.assertEqual((code_b, err_b), (EXIT_OK, ""))
        self.assertEqual(out_a, out_b)
        self.assertTrue(json.loads(out_a)["identical"])

    def test_compare_flags_numeric_differences_from_a_controlled_stand_in(self) -> None:
        edges = self.write_edges([("a", "b", 1), ("b", "c", 2)])

        def stand_in(graph: Graph, source: str) -> tuple[dict[str, float], dict[str, str | None]]:
            # Deliberately unsorted, with b and c altered but a left agreeing with Dijkstra.
            return ({"c": 9.0, "a": 0.0, "b": 1.5}, {"a": None, "b": "a", "c": "b"})

        with unittest.mock.patch("graphtk.cli.bellman_ford", stand_in):
            code, out, err = run_cli(["compare", "--edges", edges, "--directed", "--source", "a"])
        self.assertEqual((code, err), (EXIT_NEGATIVE, ""))
        document = json.loads(out)
        self.assertFalse(document["identical"])
        self.assertEqual(document["nodesCompared"], 3)
        self.assertEqual(document["dijkstraReachable"], 3)
        self.assertEqual(document["bellmanFordReachable"], 3)
        # Only the real differences, listed stably by node name.
        self.assertEqual(
            document["differences"],
            [
                {"node": "b", "dijkstra": 1.0, "bellmanFord": 1.5},
                {"node": "c", "dijkstra": 3.0, "bellmanFord": 9.0},
            ],
        )

    def test_compare_flags_a_reachable_set_mismatch_from_a_controlled_stand_in(self) -> None:
        edges = self.write_edges([("a", "b", 1), ("b", "c", 2)])

        def stand_in(graph: Graph, source: str) -> tuple[dict[str, float], dict[str, str | None]]:
            return ({"a": 0.0, "b": 1.0}, {"a": None, "b": "a"})

        with unittest.mock.patch("graphtk.cli.bellman_ford", stand_in):
            code, out, err = run_cli(["compare", "--edges", edges, "--directed", "--source", "a"])
        self.assertEqual((code, err), (EXIT_NEGATIVE, ""))
        document = json.loads(out)
        self.assertFalse(document["identical"])
        self.assertEqual(document["nodesCompared"], 2)
        self.assertEqual(document["dijkstraReachable"], 3)
        self.assertEqual(document["bellmanFordReachable"], 2)
        # Every shared node agrees, so there is no per-node difference to list.
        self.assertNotIn("differences", document)

    def test_compare_with_negative_weights_is_a_clean_error(self) -> None:
        edges = self.write_edges([("a", "b", 1), ("b", "c", -2)])
        code, out, err = run_cli(["compare", "--edges", edges, "--directed", "--source", "a"])
        self.assertEqual(code, EXIT_ERROR)
        self.assert_error_document(out, err, "negative_weight_error")

    def test_compare_with_an_unknown_source_is_a_clean_error(self) -> None:
        edges = self.write_edges([("a", "b", 1)])
        code, out, err = run_cli(["compare", "--edges", edges, "--source", "missing"])
        self.assertEqual(code, EXIT_ERROR)
        self.assert_error_document(out, err, "validation_error")

    def test_compare_with_malformed_input_is_a_clean_error(self) -> None:
        path = os.path.join(self.directory.name, "broken.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write('{"source": "a", "target": "b"}\nnot json\n')
        code, out, err = run_cli(["compare", "--edges", path, "--source", "a"])
        self.assertEqual(code, EXIT_ERROR)
        document = self.assert_error_document(out, err, "parse_error")
        self.assertEqual(document["line"], 2)

    def test_negative_cycle_is_a_clean_error_not_a_verdict(self) -> None:
        edges = self.write_edges([("a", "b", 1), ("b", "c", -2), ("c", "b", 1)])
        code, out, err = run_cli(["bellman-ford", "--edges", edges, "--directed", "--source", "a"])
        self.assertEqual(code, EXIT_ERROR)
        self.assert_error_document(out, err, "negative_cycle_error")

    def test_negative_weights_stay_errors_for_dijkstra_and_astar_commands(self) -> None:
        edges = self.write_edges([("a", "b", 1), ("b", "c", -2)])
        heuristic = self.write_heuristic(build_graph(True, [("a", "b", 1), ("b", "c", -2)]))
        code, out, err = run_cli(["dijkstra", "--edges", edges, "--directed", "--source", "a"])
        self.assertEqual(code, EXIT_ERROR)
        self.assert_error_document(out, err, "negative_weight_error")
        code, out, err = run_cli(["astar", "--edges", edges, "--directed", "--source", "a", "--target", "c", "--heuristic", heuristic])
        self.assertEqual(code, EXIT_ERROR)
        self.assert_error_document(out, err, "negative_weight_error")

    def test_astar_with_an_unreachable_target_is_a_clean_error(self) -> None:
        rows = [("a", "b", 3), ("c", "d", 2)]
        edges = self.write_edges(rows)
        heuristic = self.write_heuristic(build_graph(True, rows))
        code, out, err = run_cli(["astar", "--edges", edges, "--directed", "--source", "a", "--target", "c", "--heuristic", heuristic])
        self.assertEqual(code, EXIT_ERROR)
        self.assert_error_document(out, err, "validation_error")


if __name__ == "__main__":
    unittest.main()
