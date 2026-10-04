# graph-computation-toolkit

Graph construction, traversal, shortest paths, components, ordering, ranking, clustering and
incremental components (Python standard library only).

## Install and entry point

```
python3 -m pip install -e .
graph-computation-toolkit --help
graph-computation-toolkit describe
```

Package `graphtk`; console script `graph-computation-toolkit`. Each subcommand reads a **JSONL edge
list** and writes **one JSON document to stdout**; a failure writes one JSON document to stderr.

## Input

One JSON object per line, blank lines and `#` comments ignored:

```json
{"source": "a", "target": "b", "weight": 2.5}
```

`weight` is optional (default `1.0`), must be a finite `int` or `float` — booleans, strings, `null`
and the non-standard `NaN`/`Infinity`/`-Infinity` constants are rejected — and may be negative for
the algorithms that support it. Unknown fields are rejected, and every parse error carries its
`line`.

Graphs are **undirected unless `--directed` is passed**.

`astar` additionally reads `--heuristic`: a UTF-8 JSON object mapping node name to an estimated
remaining cost, e.g. `{"a": 3, "b": 2, "c": 1}`. Missing nodes count as `0`. Values must be finite
non-negative numbers (booleans, strings, `null` and the `NaN`/`Infinity` constants are rejected),
keys must name graph nodes, the target must be `0`, and the heuristic must be consistent —
`h(u) <= weight(u, v) + h(v)` on every walkable edge; the first violating edge in node order is
reported. An empty object degenerates to Dijkstra.

## Commands

| Command | Purpose | Exit codes |
|---|---|---|
| `describe` | capabilities, edge fields, exit codes | 0 |
| `stats` | nodes, edges, self-loops, negative weights, component count | 0 / **3** (empty) |
| `bfs --source S` | hop distances plus a depth-first pre-order | 0 / **3** / 2 |
| `dijkstra --source S [--target T]` | non-negative shortest paths; `--target` adds `path` | 0 / **3** / 2 |
| `bellman-ford --source S [--target T]` | shortest paths with negative weights | 0 / **3** / 2 |
| `astar --source S --target T --heuristic H` | A* shortest path to one target; prints `distance`, `path`, `expanded` | 0 / 2 |
| `components` | weakly connected components, each sorted | 0 / **3** / 2 |
| `toposort` | topological order (**requires `--directed`**) | 0 / 2 |
| `pagerank [--damping] [--tolerance]` | power iteration with explicit convergence reporting | 0 / **3** (not converged) / 2 |
| `centrality` | normalised degree centrality | 0 / **3** / 2 |
| `clustering` | local clustering coefficients, average, transitivity, triangle and wedge counts | 0 / **3** (empty) / 2 |
| `compare --source S` | run Dijkstra **and** Bellman-Ford and check they agree | 0 / **3** (disagree) / 2 |

`--edges -` reads from stdin.

## What the results promise

* **Determinism** — every traversal, relaxation and iteration walks nodes in sorted order, so the same
  input always produces byte-identical output.
* **Unreachable is absent, never invented.** BFS and both shortest-path routines simply omit nodes they
  cannot reach; no `-1`, no `inf` standing in for "unknown".
* **A path is consistent with its distance.** `path_from` reconstructs the route, and the test suite
  re-walks it to check the edge weights add up to the reported distance.
* **Refusals carry evidence.** Dijkstra over a negative edge raises `negative_weight_error` naming the
  edge; Bellman-Ford on a negative cycle raises `negative_cycle_error`; a cyclic graph under `toposort`
  raises `cycle_error` with the nodes that could never be released.
* **A\* is Dijkstra with a promise.** `astar` accepts only non-negative graphs and a heuristic that
  is finite, non-negative, zero at the target and consistent (`h(u) <= w(u,v) + h(v)`), so the
  reported distance and path are exactly Dijkstra's while `expanded` counts the distinct nodes the
  heuristic actually pruned the search to; an empty heuristic expands the same nodes Dijkstra does.
  A negative edge raises `negative_weight_error`; an unknown node, a non-zero target, an
  inconsistent value (naming the first violating edge in node order) or an unreachable target raise
  `validation_error`; a malformed heuristic file raises `parse_error`. All errors exit **2** and
  write nothing to stdout.
* **Two implementations must agree.** `compare` is the guarantee: Dijkstra and Bellman-Ford are
  independent code paths, and the command reports `identical`, the node count compared, both reachable
  counts, and every differing node — exiting **3** when they disagree.
* **Clustering is measured on one simple undirected view.** `clustering` returns each node's local
  coefficient (edges between its distinct neighbours over `k(k−1)/2`, `0` when `k < 2`), the average
  over **all** nodes, the transitivity (`3·triangles / connectedTriples`, `0` when the denominator is
  zero), and the triangle and connected-triple counts. An edge in either direction counts once, mutual
  arcs and duplicate input still make a single adjacency, weights are ignored, and a self-loop is never
  a neighbour or a triangle edge — so the document is identical for the same edges in any order.
* **Incremental labelling is honest.** `IncrementalComponents` merges components on insertion in
  near-constant time; a **removal** cannot be undone by union-find, so the structure marks itself stale,
  names the affected nodes, and makes `labels()` fail until `recompute()` is called instead of returning
  a quietly wrong answer. No insertion clears that state — not an edge between unrelated components,
  not an edge with a new node, and not re-adding the edge just removed — because one merge can never
  vouch for the whole graph; only `recompute()` rebuilds from the graph as it currently stands.

## Layout

```
graphtk/graph.py        graph, edges, CSR, degrees, diagnostics
graphtk/algorithms.py   BFS/DFS, Dijkstra, A*, Bellman-Ford, components, topological sort, PageRank, centrality, clustering
graphtk/incremental.py  union-find components with staleness tracking
graphtk/cli.py          thirteen subcommands and the exit-code contract
tests/                  structure, traversal, shortest paths, ordering, ranking, clustering, incremental behaviour
```
