"""A* search: public function contract, heuristic validation, CLI command and determinism."""

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


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def sample_directed() -> Graph:
    graph = Graph(directed=True)
    graph.add_edge("a", "b", 1.0)
    graph.add_edge("b", "c", 2.0)
    graph.add_edge("a", "c", 5.0)
    graph.add_edge("c", "d", 1.0)
    return graph


class AstarFunctionTests(unittest.TestCase):
    def test_finds_the_cheapest_route_and_rebuilds_the_path(self) -> None:
        distance, previous, expanded = astar(sample_directed(), "a", "d", {"a": 4, "b": 3, "c": 1, "d": 0})
        self.assertEqual(distance, 4.0)
        self.assertEqual(path_from(previous, "d"), ["a", "b", "c", "d"])
        self.assertEqual(expanded, 4)

    def test_empty_heuristic_matches_dijkstra_in_distance_and_path(self) -> None:
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
        dj_distance, dj_previous = dijkstra(graph, "a")
        for target in graph.nodes():
            if target == "a":
                continue
            distance, previous, _ = astar(graph, "a", target, {})
            self.assertEqual(distance, dj_distance[target])
            self.assertEqual(path_from(previous, target), path_from(dj_previous, target))

    def test_undirected_graph_searches_both_directions(self) -> None:
        graph = Graph()
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("b", "c", 2.0)
        graph.add_edge("a", "c", 5.0)
        distance, previous, _ = astar(graph, "c", "a", {"a": 0, "b": 1, "c": 3})
        self.assertEqual(distance, 3.0)
        self.assertEqual(path_from(previous, "a"), ["c", "b", "a"])

    def test_zero_weight_edges(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 0.0)
        graph.add_edge("b", "c", 0.0)
        graph.add_edge("a", "c", 3.0)
        distance, previous, _ = astar(graph, "a", "c", {})
        self.assertEqual(distance, 0.0)
        self.assertEqual(path_from(previous, "c"), ["a", "b", "c"])

    def test_source_equals_target_is_a_zero_length_path(self) -> None:
        distance, previous, expanded = astar(sample_directed(), "a", "a", {})
        self.assertEqual(distance, 0.0)
        self.assertEqual(path_from(previous, "a"), ["a"])
        self.assertEqual(expanded, 1)

    def test_a_guiding_heuristic_skips_the_expensive_branch(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("a", "c", 100.0)
        graph.add_edge("b", "t", 1.0)
        graph.add_edge("c", "t", 1.0)
        _, previous, expanded = astar(graph, "a", "t", {"a": 2, "b": 1, "c": 1, "t": 0})
        self.assertEqual(path_from(previous, "t"), ["a", "b", "t"])
        self.assertEqual(expanded, 3)  # c is never expanded

    def test_equal_cost_paths_break_ties_by_node_name(self) -> None:
        graph = Graph(directed=True)
        for target in ("b", "c"):
            graph.add_edge("a", target, 1.0)
            graph.add_edge(target, "z", 1.0)
        distance, previous, _ = astar(graph, "a", "z", {})
        self.assertEqual(distance, 2.0)
        self.assertEqual(path_from(previous, "z"), ["a", "b", "z"])

    def test_negative_edge_is_refused_with_evidence(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", -1.0)
        with self.assertRaises(NegativeWeightError) as caught:
            astar(graph, "a", "b", {})
        self.assertEqual(caught.exception.context["edge"], "a->b")
        self.assertEqual(caught.exception.context["weight"], -1.0)

    def test_unknown_source_and_target_are_validation_errors(self) -> None:
        graph = sample_directed()
        with self.assertRaises(ValidationError):
            astar(graph, "nope", "d", {})
        with self.assertRaises(ValidationError):
            astar(graph, "a", "nope", {})

    def test_unreachable_target_raises_instead_of_inventing_a_distance(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1.0)
        graph.add_node("z")
        with self.assertRaises(ValidationError) as caught:
            astar(graph, "a", "z", {})
        self.assertIn("unreachable target", caught.exception.message)

    def test_heuristic_value_must_be_a_finite_non_negative_number(self) -> None:
        graph = sample_directed()
        for bad in (True, float("nan"), float("inf"), -1.0, "1", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    astar(graph, "a", "d", {"b": bad})  # type: ignore[dict-item]

    def test_heuristic_cannot_name_nodes_outside_the_graph(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            astar(sample_directed(), "a", "d", {"zzz": 1})
        self.assertIn("zzz", caught.exception.message)

    def test_target_estimate_must_be_zero(self) -> None:
        with self.assertRaises(ValidationError):
            astar(sample_directed(), "a", "d", {"d": 1})

    def test_inconsistent_heuristic_reports_the_first_violating_edge_in_sorted_order(self) -> None:
        graph = Graph(directed=True)
        graph.add_edge("a", "z", 1.0)
        graph.add_edge("b", "z", 1.0)
        with self.assertRaises(ValidationError) as caught:
            astar(graph, "a", "z", {"a": 5, "b": 4, "z": 0})
        self.assertEqual(caught.exception.context["edge"], "a->z")

    def test_consistency_is_checked_along_every_traversable_direction(self) -> None:
        graph = Graph()  # undirected: the a-b edge is walkable both ways
        graph.add_edge("a", "b", 1.0)
        with self.assertRaises(ValidationError) as caught:
            astar(graph, "b", "a", {"a": 0, "b": 5})
        self.assertEqual(caught.exception.context["edge"], "b->a")

    def test_heuristic_key_order_and_edge_order_do_not_change_the_result(self) -> None:
        rows = [("a", "b", 2.0), ("b", "c", 1.0), ("a", "c", 9.0), ("c", "d", 2.0), ("b", "d", 5.0)]
        heuristic = {"a": 4, "b": 3, "c": 2, "d": 0}

        def build(order: list[int]) -> Graph:
            graph = Graph(directed=True)
            for index in order:
                graph.add_edge(*rows[index])
            return graph

        reference = astar(build(list(range(len(rows)))), "a", "d", heuristic)
        shuffled = astar(build([3, 0, 4, 1, 2]), "a", "d", dict(reversed(list(heuristic.items()))))
        self.assertEqual(reference, shuffled)


class AstarCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.edges = os.path.join(self.directory.name, "edges.jsonl")
        rows = [
            {"source": "a", "target": "b", "weight": 1.0},
            {"source": "b", "target": "c", "weight": 2.0},
            {"source": "a", "target": "c", "weight": 5.0},
            {"source": "c", "target": "d", "weight": 1.0},
        ]
        with open(self.edges, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        self.heuristic = self.write_heuristic({"a": 4, "b": 3, "c": 1, "d": 0})

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_heuristic(self, mapping: dict, name: str = "heuristic.json") -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(mapping))
        return path

    def write_raw(self, text: str, name: str) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return path

    def run_astar(self, *extra: str) -> tuple[int, str, str]:
        return run_cli(["astar", "--edges", self.edges, "--directed", "--source", "a", "--target", "d", *extra])

    def test_success_reports_distance_path_and_expanded(self) -> None:
        code, out, err = self.run_astar("--heuristic", self.heuristic)
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertEqual(
            json.loads(out),
            {"source": "a", "target": "d", "distance": 4.0, "path": ["a", "b", "c", "d"], "expanded": 4},
        )

    def test_empty_heuristic_agrees_with_dijkstra(self) -> None:
        empty = self.write_heuristic({}, "empty.json")
        code, out, err = self.run_astar("--heuristic", empty)
        self.assertEqual((code, err), (EXIT_OK, ""))
        astar_document = json.loads(out)
        code, out, _ = run_cli(["dijkstra", "--edges", self.edges, "--directed", "--source", "a", "--target", "d"])
        self.assertEqual(code, EXIT_OK)
        dijkstra_document = json.loads(out)
        self.assertEqual(astar_document["distance"], dijkstra_document["distance"])
        self.assertEqual(astar_document["path"], dijkstra_document["path"])

    def test_output_bytes_do_not_depend_on_edge_or_heuristic_order(self) -> None:
        rows = [
            {"source": "c", "target": "d", "weight": 1.0},
            {"source": "a", "target": "c", "weight": 5.0},
            {"source": "b", "target": "c", "weight": 2.0},
            {"source": "a", "target": "b", "weight": 1.0},
        ]
        reordered_edges = self.write_raw("".join(json.dumps(row) + "\n" for row in rows), "reordered.jsonl")
        reordered_heuristic = self.write_raw('{"d": 0, "c": 1, "b": 3, "a": 4}', "reordered.json")
        _, first, _ = self.run_astar("--heuristic", self.heuristic)
        code, second, _ = run_cli(
            ["astar", "--edges", reordered_edges, "--directed", "--source", "a", "--target", "d", "--heuristic", reordered_heuristic]
        )
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(first, second)

    def test_unreachable_target_is_a_validation_error_without_partial_output(self) -> None:
        code, out, err = self.run_astar("--heuristic", self.write_heuristic({"a": 0, "b": 0, "c": 0, "d": 0}))
        self.assertEqual(code, EXIT_OK)  # d is reachable here; sanity check the fixture
        self.assertEqual(json.loads(out)["path"], ["a", "b", "c", "d"])
        split = self.write_raw('{"source": "a", "target": "b"}\n{"source": "c", "target": "d"}\n', "split.jsonl")
        heuristic = self.write_heuristic({"a": 0, "b": 0, "c": 0, "d": 0}, "split.json")
        code, out, err = run_cli(["astar", "--edges", split, "--directed", "--source", "a", "--target", "d", "--heuristic", heuristic])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        document = json.loads(err)
        self.assertEqual(document["error"], "validation_error")
        self.assertIn("unreachable target", document["message"])

    def test_negative_edge_is_a_negative_weight_error(self) -> None:
        negative = self.write_raw('{"source": "a", "target": "b", "weight": -1}\n', "negative.jsonl")
        code, out, err = run_cli(["astar", "--edges", negative, "--source", "a", "--target", "b", "--heuristic", self.heuristic])
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        document = json.loads(err)
        self.assertEqual(document["error"], "negative_weight_error")
        self.assertEqual(document["edge"], "a->b")

    def test_malformed_heuristic_files_are_parse_errors(self) -> None:
        cases = {
            "not json": "{oops}",
            "top level array": "[1, 2]",
            "boolean value": '{"a": true}',
            "NaN value": '{"a": NaN}',
            "Infinity value": '{"a": Infinity}',
            "string value": '{"a": "soon"}',
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                path = self.write_raw(text, f"bad-{name.replace(' ', '-')}.json")
                code, out, err = self.run_astar("--heuristic", path)
                self.assertEqual(code, EXIT_ERROR)
                self.assertEqual(out, "")
                self.assertEqual(json.loads(err)["error"], "parse_error")

    def test_invalid_heuristic_content_is_a_validation_error(self) -> None:
        cases = {
            "unknown node": {"zzz": 1},
            "target not zero": {"d": 2},
            "negative estimate": {"b": -1},
            "inconsistent": {"a": 9, "b": 0, "c": 0, "d": 0},
        }
        for name, mapping in cases.items():
            with self.subTest(name=name):
                path = self.write_heuristic(mapping, f"invalid-{name.replace(' ', '-')}.json")
                code, out, err = self.run_astar("--heuristic", path)
                self.assertEqual(code, EXIT_ERROR)
                self.assertEqual(out, "")
                self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_missing_heuristic_file_is_an_error_document(self) -> None:
        code, out, err = self.run_astar("--heuristic", os.path.join(self.directory.name, "missing.json"))
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        self.assertEqual(json.loads(err)["error"], "validation_error")

    def test_describe_lists_the_astar_subcommand(self) -> None:
        code, out, _ = run_cli(["describe"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("astar", json.loads(out)["subcommands"])


if __name__ == "__main__":
    unittest.main()
