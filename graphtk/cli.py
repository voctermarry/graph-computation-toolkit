"""Command line entry point.

Every subcommand reads a JSONL edge list and writes one JSON document to stdout; errors go to stderr
as a single JSON document. Exit codes: 0 success, 2 input/usage error, 3 a report was produced whose
verdict is negative (an empty answer, a refresh that had to repair the log, or two algorithms that
disagree).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, NoReturn, Sequence

from . import __version__
from .algorithms import (
    astar,
    bellman_ford,
    bfs,
    clustering,
    components,
    degree_centrality,
    dfs,
    dijkstra,
    pagerank,
    path_from,
    topological_sort,
)
from .errors import GraphError, ParseError, ValidationError
from .graph import Edge, Graph

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_NEGATIVE = 3


def canonical(document: dict[str, Any]) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _emit(document: dict[str, Any]) -> None:
    sys.stdout.write(canonical(document) + "\n")


def _reject_constant(text: str) -> NoReturn:
    # json.loads accepts the non-standard tokens NaN, Infinity and -Infinity by default, producing
    # float values that cannot be serialised back to strict JSON and would poison every computation.
    # Refuse them while parsing so they are reported as a parse error on the line they occupy.
    raise ValueError(f"non-standard numeric constant: {text}")


def _read_edges(path: str, directed: bool) -> Graph:
    lines = sys.stdin.read().splitlines() if path == "-" else _read_file(path)
    graph = Graph(directed=directed)
    for number, text in enumerate(lines, start=1):
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            document = json.loads(stripped, parse_constant=_reject_constant)
        except json.JSONDecodeError as error:
            raise ParseError(f"invalid JSON: {error.msg}", line=number) from error
        except ValueError as error:
            raise ParseError(str(error), line=number) from error
        if not isinstance(document, dict):
            raise ParseError("each line must be a JSON object", line=number)
        unknown = sorted(set(document) - {"source", "target", "weight"})
        if unknown:
            raise ParseError(f"unknown field(s): {', '.join(unknown)}", line=number)
        if "source" not in document or "target" not in document:
            raise ParseError("each line needs source and target", line=number)
        source, target = document["source"], document["target"]
        if not isinstance(source, str) or not isinstance(target, str) or not source or not target:
            raise ParseError("source and target must be non-empty strings", line=number)
        weight = document.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise ParseError("weight must be a finite number", line=number)
        graph.add_edge(source, target, float(weight))
    return graph


def _read_file(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().splitlines()
    except OSError as error:
        raise ValidationError(f"cannot read edges: {error.strerror or error}", value=path) from error


def _read_heuristic(path: str) -> dict[str, float]:
    """Read the node -> remaining-cost object; malformed files are ParseError, never partial data."""
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        raise ValidationError(f"cannot read heuristic: {error.strerror or error}", value=path) from error
    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ParseError(f"invalid JSON: {error.msg}") from error
    except ValueError as error:
        raise ParseError(str(error)) from error
    if not isinstance(document, dict):
        raise ParseError("heuristic must be a JSON object of node name to remaining cost")
    values: dict[str, float] = {}
    for node in sorted(document):
        raw = document[node]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ParseError(f"heuristic value for {node!r} must be a finite number")
        values[node] = float(raw)
    return values


def _round(distances: dict[str, float]) -> dict[str, float]:
    return {node: round(value, 10) for node, value in sorted(distances.items())}


# -- commands ------------------------------------------------------------------------------------
def _command_describe(_: argparse.Namespace) -> int:
    _emit(
        {
            "name": "graph-computation-toolkit",
            "version": __version__,
            "subcommands": [
                "astar",
                "bellman-ford",
                "bfs",
                "centrality",
                "clustering",
                "compare",
                "components",
                "describe",
                "dijkstra",
                "dfs",
                "pagerank",
                "stats",
                "toposort",
            ],
            "edgeFields": ["source", "target", "weight"],
            "directedByDefault": False,
            "deterministic": "every traversal and iteration uses sorted node order",
            "exitCodes": {"ok": EXIT_OK, "error": EXIT_ERROR, "negativeVerdict": EXIT_NEGATIVE},
        }
    )
    return EXIT_OK


def _graph_from(args: argparse.Namespace) -> Graph:
    return _read_edges(args.edges, args.directed)


def _command_stats(args: argparse.Namespace) -> int:
    graph = _graph_from(args)
    document = graph.to_document()
    document["components"] = len(components(graph))
    _emit(document)
    return EXIT_OK if graph.node_count else EXIT_NEGATIVE


def _command_bfs(args: argparse.Namespace) -> int:
    graph = _graph_from(args)
    distances = bfs(graph, args.source)
    _emit({"source": args.source, "reachable": len(distances), "distances": distances, "order": dfs(graph, args.source)})
    return EXIT_OK if len(distances) > 1 else EXIT_NEGATIVE


def _command_dijkstra(args: argparse.Namespace) -> int:
    graph = _graph_from(args)
    distance, previous = dijkstra(graph, args.source)
    document: dict[str, Any] = {"source": args.source, "distances": _round(distance), "reachable": len(distance)}
    if args.target:
        document["target"] = args.target
        document["path"] = path_from(previous, args.target)
        document["distance"] = round(distance[args.target], 10)
    _emit(document)
    return EXIT_OK if args.target in distance or (args.target is None and len(distance) > 1) else EXIT_NEGATIVE


def _command_astar(args: argparse.Namespace) -> int:
    graph = _graph_from(args)
    heuristic = _read_heuristic(args.heuristic)
    distance, previous, expanded = astar(graph, args.source, args.target, heuristic)
    _emit(
        {
            "source": args.source,
            "target": args.target,
            "distance": round(distance[args.target], 10),
            "path": path_from(previous, args.target),
            "expanded": expanded,
        }
    )
    return EXIT_OK


def _command_bellman_ford(args: argparse.Namespace) -> int:
    graph = _graph_from(args)
    distance, previous = bellman_ford(graph, args.source)
    document: dict[str, Any] = {"source": args.source, "distances": _round(distance), "reachable": len(distance)}
    if args.target:
        document["target"] = args.target
        document["path"] = path_from(previous, args.target)
    _emit(document)
    return EXIT_OK if len(distance) > 1 else EXIT_NEGATIVE


def _command_components(args: argparse.Namespace) -> int:
    graph = _graph_from(args)
    groups = components(graph)
    _emit({"components": groups, "count": len(groups)})
    return EXIT_OK if groups else EXIT_NEGATIVE


def _command_toposort(args: argparse.Namespace) -> int:
    if not args.directed:
        raise ValidationError("toposort requires --directed")
    graph = _graph_from(args)
    order = topological_sort(graph)
    _emit({"order": order, "count": len(order)})
    return EXIT_OK


def _command_pagerank(args: argparse.Namespace) -> int:
    graph = _graph_from(args)
    result = pagerank(graph, damping=args.damping, tolerance=args.tolerance)
    document = result.to_document()
    document["damping"] = args.damping
    _emit(document)
    return EXIT_OK if result.converged else EXIT_NEGATIVE


def _command_centrality(args: argparse.Namespace) -> int:
    graph = _graph_from(args)
    _emit({"centrality": degree_centrality(graph)})
    return EXIT_OK if graph.node_count else EXIT_NEGATIVE


def _command_clustering(args: argparse.Namespace) -> int:
    graph = _graph_from(args)
    _emit(clustering(graph).to_document())
    return EXIT_OK if graph.node_count else EXIT_NEGATIVE


def _command_compare(args: argparse.Namespace) -> int:
    """Two independent shortest-path implementations over the same graph must agree exactly."""
    graph = _graph_from(args)
    dijkstra_distance, _ = dijkstra(graph, args.source)
    bellman_distance, _ = bellman_ford(graph, args.source)
    shared = sorted(set(dijkstra_distance) & set(bellman_distance))
    differences = [
        {"node": node, "dijkstra": round(dijkstra_distance[node], 10), "bellmanFord": round(bellman_distance[node], 10)}
        for node in shared
        if abs(dijkstra_distance[node] - bellman_distance[node]) > 1e-9
    ]
    identical = not differences and set(dijkstra_distance) == set(bellman_distance)
    document: dict[str, Any] = {
        "source": args.source,
        "identical": identical,
        "nodesCompared": len(shared),
        "dijkstraReachable": len(dijkstra_distance),
        "bellmanFordReachable": len(bellman_distance),
    }
    if differences:
        document["differences"] = differences
    _emit(document)
    return EXIT_OK if identical else EXIT_NEGATIVE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graph-computation-toolkit", description="graph algorithms over a JSONL edge list")
    parser.add_argument("--version", action="version", version=f"graph-computation-toolkit {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("describe", help="print capabilities as JSON").set_defaults(handler=_command_describe)

    def with_graph(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--edges", required=True, help="JSONL edges, or - for stdin")
        sub.add_argument("--directed", action="store_true", help="treat edges as one-way")
        return sub

    with_graph("stats", "graph counters").set_defaults(handler=_command_stats)

    bfs_command = with_graph("bfs", "hop distances and a depth-first order")
    bfs_command.add_argument("--source", required=True)
    bfs_command.set_defaults(handler=_command_bfs)

    dijkstra_command = with_graph("dijkstra", "non-negative shortest paths")
    dijkstra_command.add_argument("--source", required=True)
    dijkstra_command.add_argument("--target")
    dijkstra_command.set_defaults(handler=_command_dijkstra)

    astar_command = with_graph("astar", "A* shortest path to one target with a heuristic")
    astar_command.add_argument("--source", required=True)
    astar_command.add_argument("--target", required=True)
    astar_command.add_argument("--heuristic", required=True, help="UTF-8 JSON object of node to remaining cost")
    astar_command.set_defaults(handler=_command_astar)

    bellman_command = with_graph("bellman-ford", "shortest paths with negative weights")
    bellman_command.add_argument("--source", required=True)
    bellman_command.add_argument("--target")
    bellman_command.set_defaults(handler=_command_bellman_ford)

    with_graph("components", "weakly connected components").set_defaults(handler=_command_components)
    with_graph("toposort", "topological order (needs --directed)").set_defaults(handler=_command_toposort)

    pagerank_command = with_graph("pagerank", "power-iteration PageRank")
    pagerank_command.add_argument("--damping", type=float, default=0.85)
    pagerank_command.add_argument("--tolerance", type=float, default=1e-9)
    pagerank_command.set_defaults(handler=_command_pagerank)

    with_graph("centrality", "normalised degree centrality").set_defaults(handler=_command_centrality)

    with_graph("clustering", "local clustering coefficients, average and transitivity").set_defaults(handler=_command_clustering)

    compare_command = with_graph("compare", "run two shortest-path algorithms and check they agree")
    compare_command.add_argument("--source", required=True)
    compare_command.set_defaults(handler=_command_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except GraphError as error:
        sys.stderr.write(canonical(error.to_document()) + "\n")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
