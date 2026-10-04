"""Shortest-path consistency regression: Dijkstra, Bellman-Ford and A* must agree.

Every non-negative fixture graph is solved through the public Python entry points and through the
CLI, and all three algorithms must report the same distance for every reachable node. A* always
runs with an all-zero heuristic, where it must match Dijkstra's target distance exactly. Routes are
not compared between algorithms -- equal-length alternatives may be chosen differently -- but every
returned route is checked to start at the source, end at the target, follow real traversable edges,
and sum to the reported distance. All weights are finite integers so a float rounding artifact can
never masquerade as an algorithmic disagreement.
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


def oracle_distances(graph: Graph, source: str) -> dict[str, float]:
    """Independent distance oracle: the minimum cost over every simple path from `source`.

    Enumerating simple paths is exponential but exact, and shares no code with the relaxation
    loops under test -- so agreement is evidence, not coincidence. Only valid on the small
    fixtures here, and only because none of them contains a negative cycle.
    """
    best: dict[str, float] = {source: 0.0}
    stack = [(source, 0.0, {source})]
    while stack:
        node, cost, visited = stack.pop()
        if node not in best or cost < best[node]:
            best[node] = cost
        for neighbour, weight in graph.neighbors(node):
            if neighbour not in visited:
                stack.append((neighbour, cost + weight, visited | {neighbour}))
    return best


def graph_cases() -> list[tuple[str, Graph, str]]:
    """Non-negative integer-weighted fixtures covering the public graph semantics."""
    cases: list[tuple[str, Graph, str]] = []

    # Directed and connected, with two equal-length cheapest routes to d (a-b-d and a-c-d).
    equal_routes = Graph(directed=True)
    for source, target, weight in (("a", "b", 1), ("b", "d", 2), ("a", "c", 2), ("c", "d", 1), ("d", "e", 3), ("a", "e", 10)):
        equal_routes.add_edge(source, target, weight)
    cases.append(("directed_equal_length_routes", equal_routes, "a"))

    # Undirected, with a zero-weight edge and a positive self-loop that must never be used.
    undirected = Graph(directed=False)
    for source, target, weight in (("a", "b", 2), ("b", "c", 0), ("c", "d", 4), ("a", "d", 7), ("b", "d", 1), ("c", "c", 5)):
        undirected.add_edge(source, target, weight)
    cases.append(("undirected_zero_edge_self_loop", undirected, "a"))

    # Directed and disconnected: the source component is a chain, the rest is unreachable.
    disconnected = Graph(directed=True)
    for source, target, weight in (("a", "b", 1), ("b", "c", 2), ("d", "e", 1)):
        disconnected.add_edge(source, target, weight)
    disconnected.add_node("f")
    cases.append(("disconnected_directed", disconnected, "a"))

    # A repeated edge: the second write overwrites the first, so a->b costs 1, not 5.
    duplicates = Graph(directed=True)
    duplicates.add_edge("a", "b", 5)
    duplicates.add_edge("a", "b", 1)
    duplicates.add_edge("b", "c", 2)
    duplicates.add_edge("a", "c", 9)
    cases.append(("duplicate_edges_last_write_wins", duplicates, "a"))

    # The source is isolated: exactly one reachable node, at distance zero.
    lonely = Graph(directed=True)
    lonely.add_edge("a", "b", 1)
    lonely.add_edge("b", "a", 1)
    lonely.add_node("z")
    cases.append(("single_reachable_node", lonely, "z"))

    # A chain of zero-weight edges beats a direct expensive edge.
    zero_chain = Graph(directed=True)
    for source, target, weight in (("a", "b", 0), ("b", "c", 0), ("a", "c", 5), ("c", "d", 1)):
        zero_chain.add_edge(source, target, weight)
    cases.append(("zero_weight_chain", zero_chain, "a"))

    return cases


def negative_weight_graph() -> Graph:
    """Directed, negative edges, no negative cycle; e->f is unreachable from a."""
    graph = Graph(directed=True)
    for source, target, weight in (("a", "b", 2), ("a", "c", 4), ("b", "c", -3), ("c", "d", 1), ("b", "d", 6), ("a", "d", 9), ("e", "f", 1)):
        graph.add_edge(source, target, weight)
    return graph


def negative_cycle_graph() -> Graph:
    """The cycle b->c->b costs -1 and is reachable from a: no finite answer exists."""
    graph = Graph(directed=True)
    graph.add_edge("a", "b", 1)
    graph.add_edge("b", "c", -2)
    graph.add_edge("c", "b", 1)
    return graph


class RouteAssertion:
    """Mixin: a returned route must be a real walk whose edge weights sum to the distance."""

    def assert_route(self, graph: Graph, source: str, target: str, path: list[str], distance: float) -> None:
        self.assertGreaterEqual(len(path), 1)
        self.assertEqual(path[0], source)
        self.assertEqual(path[-1], target)
        total = 0.0
        for left, right in zip(path, path[1:]):
            neighbours = dict(graph.neighbors(left))
            self.assertIn(right, neighbours, f"{left}->{right} is not a traversable edge")
            total += neighbours[right]
        self.assertEqual(total, distance)


class NonNegativeAgreementTests(unittest.TestCase, RouteAssertion):
    """All three algorithms, plus the oracle, must report identical distances per reachable node."""

    def test_distances_agree_across_algorithms_and_oracle(self) -> None:
        for name, graph, source in graph_cases():
            with self.subTest(case=name):
                dj_distance, _ = dijkstra(graph, source)
                bf_distance, _ = bellman_ford(graph, source)
                self.assertEqual(dj_distance, bf_distance)
                self.assertEqual(dj_distance, oracle_distances(graph, source))

    def test_astar_zero_heuristic_matches_dijkstra_target_distance(self) -> None:
        for name, graph, source in graph_cases():
            dj_distance, _ = dijkstra(graph, source)
            for target in dj_distance:
                with self.subTest(case=name, target=target):
                    distance, _, _ = astar(graph, source, target, {})
                    self.assertEqual(distance, dj_distance[target])

    def test_every_returned_route_is_a_real_walk_matching_its_distance(self) -> None:
        for name, graph, source in graph_cases():
            dj_distance, dj_previous = dijkstra(graph, source)
            bf_distance, bf_previous = bellman_ford(graph, source)
            for target in dj_distance:
                with self.subTest(case=name, target=target):
                    self.assert_route(graph, source, target, path_from(dj_previous, target), dj_distance[target])
                    self.assert_route(graph, source, target, path_from(bf_previous, target), bf_distance[target])
                    as_distance, as_previous, _ = astar(graph, source, target, {})
                    self.assert_route(graph, source, target, path_from(as_previous, target), as_distance)

    def test_equal_length_routes_may_differ_but_must_stay_valid(self) -> None:
        # a-b-d and a-c-d both cost 3: the algorithms need not pick the same one, but each pick
        # must be a genuine cheapest route.
        graph = Graph(directed=True)
        for source, target, weight in (("a", "b", 1), ("b", "d", 2), ("a", "c", 2), ("c", "d", 1)):
            graph.add_edge(source, target, weight)
        dj_distance, dj_previous = dijkstra(graph, "a")
        bf_distance, bf_previous = bellman_ford(graph, "a")
        as_distance, as_previous, _ = astar(graph, "a", "d", {})
        self.assertEqual(dj_distance["d"], bf_distance["d"])
        self.assertEqual(as_distance, dj_distance["d"])
        for previous, distance in ((dj_previous, dj_distance["d"]), (bf_previous, bf_distance["d"]), (as_previous, as_distance)):
            path = path_from(previous, "d")
            self.assertIn(path, (["a", "b", "d"], ["a", "c", "d"]))
            self.assert_route(graph, "a", "d", path, distance)


class NegativeWeightTests(unittest.TestCase, RouteAssertion):
    def test_bellman_ford_matches_the_oracle_and_rebuilds_real_routes(self) -> None:
        graph = negative_weight_graph()
        distance, previous = bellman_ford(graph, "a")
        self.assertEqual(distance, oracle_distances(graph, "a"))
        self.assertEqual(distance, {"a": 0.0, "b": 2.0, "c": -1.0, "d": 0.0})
        for target in distance:
            with self.subTest(target=target):
                self.assert_route(graph, "a", target, path_from(previous, target), distance[target])

    def test_dijkstra_and_astar_refuse_negative_weights(self) -> None:
        graph = negative_weight_graph()
        with self.assertRaises(NegativeWeightError):
            dijkstra(graph, "a")
        with self.assertRaises(NegativeWeightError):
            astar(graph, "a", "d", {})

    def test_reachable_negative_cycle_raises_negative_cycle_error(self) -> None:
        with self.assertRaises(NegativeCycleError):
            bellman_ford(negative_cycle_graph(), "a")


class ValidationFailureTests(unittest.TestCase):
    def test_unknown_source_is_a_validation_error_for_every_algorithm(self) -> None:
        graph = negative_weight_graph()
        for call in (
            lambda: dijkstra(graph, "nope"),
            lambda: bellman_ford(graph, "nope"),
            lambda: astar(graph, "nope", "a", {}),
        ):
            with self.subTest(call=call):
                with self.assertRaises(ValidationError):
                    call()

    def test_path_from_raises_for_an_unreachable_target(self) -> None:
        _, graph, source = graph_cases()[2]  # disconnected_directed
        _, previous = dijkstra(graph, source)
        with self.assertRaises(ValidationError):
            path_from(previous, "d")

    def test_astar_raises_for_an_unreachable_target(self) -> None:
        _, graph, source = graph_cases()[2]  # disconnected_directed
        with self.assertRaises(ValidationError):
            astar(graph, source, "d", {})


class CompareCLITests(unittest.TestCase):
    """The compare command: agreement, byte-stable output, and honest disagreement reports."""

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

    def disconnected_rows(self) -> list[dict]:
        return [
            {"source": "a", "target": "b", "weight": 1},
            {"source": "b", "target": "c", "weight": 2},
            {"source": "a", "target": "c", "weight": 5},
            {"source": "d", "target": "e", "weight": 1},
        ]

    def test_consistent_graph_reports_identical_with_exact_counts(self) -> None:
        path = self.write_edges(self.disconnected_rows())
        code, out, err = run_cli(["compare", "--edges", path, "--directed", "--source", "a"])
        self.assertEqual((code, err), (EXIT_OK, ""))
        document = json.loads(out)
        self.assertTrue(document["identical"])
        self.assertEqual(document["nodesCompared"], 3)  # a, b, c -- d and e are unreachable
        self.assertEqual(document["dijkstraReachable"], 3)
        self.assertEqual(document["bellmanFordReachable"], 3)
        self.assertNotIn("differences", document)

    def test_stdout_is_byte_identical_when_input_order_changes(self) -> None:
        rows = self.disconnected_rows() + [
            {"source": "a", "target": "b", "weight": 1},  # exact duplicate of an earlier line
            {"source": "c", "target": "c", "weight": 2},  # self-loop
        ]
        first = self.write_edges(rows, "first.jsonl")
        second = self.write_edges(list(reversed(rows)), "second.jsonl")
        code_a, out_a, err_a = run_cli(["compare", "--edges", first, "--directed", "--source", "a"])
        code_b, out_b, err_b = run_cli(["compare", "--edges", second, "--directed", "--source", "a"])
        self.assertEqual((code_a, err_a, code_b, err_b), (EXIT_OK, "", EXIT_OK, ""))
        self.assertEqual(out_a.encode("utf-8"), out_b.encode("utf-8"))

    def test_disagreeing_distances_exit_three_and_list_only_real_differences(self) -> None:
        path = self.write_edges(self.disconnected_rows())
        dijkstra_result = ({"a": 0.0, "b": 1.0, "c": 3.0, "d": 8.0}, {"a": None})
        bellman_result = ({"a": 0.0, "b": 7.0, "c": 3.0, "d": 2.0}, {"a": None})
        with unittest.mock.patch("graphtk.cli.dijkstra", return_value=dijkstra_result), unittest.mock.patch(
            "graphtk.cli.bellman_ford", return_value=bellman_result
        ):
            code, out, err = run_cli(["compare", "--edges", path, "--directed", "--source", "a"])
        self.assertEqual((code, err), (EXIT_NEGATIVE, ""))
        document = json.loads(out)
        self.assertFalse(document["identical"])
        self.assertEqual(document["nodesCompared"], 4)
        # Only b and d really differ; a and c agree and must not appear. Order is by node name.
        self.assertEqual(
            document["differences"],
            [
                {"node": "b", "dijkstra": 1.0, "bellmanFord": 7.0},
                {"node": "d", "dijkstra": 8.0, "bellmanFord": 2.0},
            ],
        )

    def test_disagreeing_reachable_sets_exit_three_without_inventing_differences(self) -> None:
        path = self.write_edges(self.disconnected_rows())
        dijkstra_result = ({"a": 0.0, "b": 1.0}, {"a": None})
        bellman_result = ({"a": 0.0, "b": 1.0, "c": 3.0}, {"a": None})
        with unittest.mock.patch("graphtk.cli.dijkstra", return_value=dijkstra_result), unittest.mock.patch(
            "graphtk.cli.bellman_ford", return_value=bellman_result
        ):
            code, out, err = run_cli(["compare", "--edges", path, "--directed", "--source", "a"])
        self.assertEqual((code, err), (EXIT_NEGATIVE, ""))
        document = json.loads(out)
        self.assertFalse(document["identical"])
        self.assertEqual(document["nodesCompared"], 2)
        self.assertEqual(document["dijkstraReachable"], 2)
        self.assertEqual(document["bellmanFordReachable"], 3)
        # Every shared node agrees, so there is no per-node difference to list.
        self.assertNotIn("differences", document)

    def assert_error_document(self, code: int, out: str, err: str, kind: str) -> None:
        # A failure leaves no partial success behind: stdout stays empty and stderr carries
        # exactly one JSON document in the established {error, message, ...} shape.
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        self.assertEqual(err.count("\n"), 1)
        document = json.loads(err)
        self.assertEqual(document["error"], kind)
        self.assertIn("message", document)

    def test_negative_weights_keep_the_existing_exit_two_contract(self) -> None:
        path = self.write_edges([{"source": "a", "target": "b", "weight": -1}], "negative.jsonl")
        heuristic = os.path.join(self.directory.name, "heuristic.json")
        with open(heuristic, "w", encoding="utf-8") as handle:
            handle.write("{}")
        commands = [
            ["compare", "--source", "a"],
            ["dijkstra", "--source", "a"],
            ["astar", "--source", "a", "--target", "b", "--heuristic", heuristic],
        ]
        for command in commands:
            with self.subTest(command=command[0]):
                code, out, err = run_cli([*command, "--edges", path, "--directed"])
                self.assert_error_document(code, out, err, "negative_weight_error")

    def test_negative_cycle_is_an_error_document_not_a_verdict(self) -> None:
        path = self.write_edges(
            [
                {"source": "a", "target": "b", "weight": 1},
                {"source": "b", "target": "c", "weight": -2},
                {"source": "c", "target": "b", "weight": 1},
            ],
            "cycle.jsonl",
        )
        code, out, err = run_cli(["bellman-ford", "--edges", path, "--directed", "--source", "a"])
        self.assert_error_document(code, out, err, "negative_cycle_error")

    def test_unknown_source_and_unreachable_target_are_exit_two(self) -> None:
        path = self.write_edges(self.disconnected_rows())
        code, out, err = run_cli(["compare", "--edges", path, "--directed", "--source", "nope"])
        self.assert_error_document(code, out, err, "validation_error")
        heuristic = os.path.join(self.directory.name, "heuristic.json")
        with open(heuristic, "w", encoding="utf-8") as handle:
            handle.write("{}")
        code, out, err = run_cli(["astar", "--edges", path, "--directed", "--source", "a", "--target", "d", "--heuristic", heuristic])
        self.assert_error_document(code, out, err, "validation_error")

    def test_malformed_input_is_a_parse_error_document(self) -> None:
        path = self.write_edges([{"source": "a", "target": "b"}], "broken.jsonl")
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write("{oops}\n")
        code, out, err = run_cli(["compare", "--edges", path, "--directed", "--source", "a"])
        self.assert_error_document(code, out, err, "parse_error")

    def test_cli_astar_zero_heuristic_matches_cli_dijkstra_distance(self) -> None:
        path = self.write_edges(self.disconnected_rows())
        heuristic = os.path.join(self.directory.name, "heuristic.json")
        with open(heuristic, "w", encoding="utf-8") as handle:
            handle.write("{}")
        code_d, out_d, err_d = run_cli(["dijkstra", "--edges", path, "--directed", "--source", "a", "--target", "c"])
        code_a, out_a, err_a = run_cli(["astar", "--edges", path, "--directed", "--source", "a", "--target", "c", "--heuristic", heuristic])
        self.assertEqual((code_d, err_d, code_a, err_a), (EXIT_OK, "", EXIT_OK, ""))
        dijkstra_document = json.loads(out_d)
        astar_document = json.loads(out_a)
        self.assertEqual(astar_document["distance"], dijkstra_document["distance"])
        self.assertEqual(astar_document["distance"], 3.0)
        # The reported route is a real walk: consecutive nodes are adjacent and weights sum to 3.
        route = astar_document["path"]
        self.assertEqual((route[0], route[-1]), ("a", "c"))
        weights = {("a", "b"): 1, ("b", "c"): 2, ("a", "c"): 5}
        self.assertEqual(sum(weights[(left, right)] for left, right in zip(route, route[1:])), 3)


if __name__ == "__main__":
    unittest.main()
