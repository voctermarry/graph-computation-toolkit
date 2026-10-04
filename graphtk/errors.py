"""Exception hierarchy with stable `kind` values.

The three failure modes that matter for graph work each get their own type, because each one means
"the answer you asked for does not exist" rather than "the call was malformed": a negative edge under
Dijkstra, a negative cycle under Bellman-Ford, and a cycle under topological sort.
"""

from __future__ import annotations


class GraphError(Exception):
    """Base class for every error this package raises on purpose."""

    kind = "graph_error"

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = {key: value for key, value in context.items() if value is not None}

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {"error": self.kind, "message": self.message}
        document.update(self.context)
        return document


class ParseError(GraphError):
    """Malformed input rows or documents."""

    kind = "parse_error"

    def __init__(self, message: str, *, line: int | None = None, **context: object) -> None:
        super().__init__(message, line=line, **context)


class ValidationError(GraphError):
    """A request that is well-formed but not allowed: unknown node, undirected sort, bad damping."""

    kind = "validation_error"


class NegativeWeightError(GraphError):
    """Dijkstra was asked for shortest paths over a graph with a negative edge."""

    kind = "negative_weight_error"


class NegativeCycleError(GraphError):
    """Bellman-Ford found a negative cycle reachable from the source: no finite answer exists."""

    kind = "negative_cycle_error"


class CycleError(GraphError):
    """A topological order was requested for a graph that has a directed cycle."""

    kind = "cycle_error"


class OutputError(GraphError):
    """The output target cannot be used safely."""

    kind = "output_error"
