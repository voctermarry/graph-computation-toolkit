# graph-computation-toolkit

Graph construction, traversal, shortest paths, components, ordering, ranking and incremental
components (Python standard library only).

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

`weight` is optional (default `1.0`) and may be negative for the algorithms that support it. Unknown
fields are rejected, and every parse error carries its `line`.

Graphs are **undirected unless `--directed` is passed**.

## Commands

| Command | Purpose | Exit codes |
|---|---|---|
| `describe` | capabilities, edge fields, exit codes | 0 |
| `stats` | nodes, edges, self-loops, negative weights, component count | 0 / **3** (empty) |
| `bfs --source S` | hop distances plus a depth-first pre-order | 0 / **3** / 2 |
| `dijkstra --source S [--target T]` | non-negative shortest paths; `--target` adds `path` | 0 / **3** / 2 |
| `bellman-ford --source S [--target T]` | shortest paths with negative weights | 0 / **3** / 2 |
| `components` | weakly connected components, each sorted | 0 / **3** / 2 |
| `toposort` | topological order (**requires `--directed`**) | 0 / 2 |
| `pagerank [--damping] [--tolerance]` | power iteration with explicit convergence reporting | 0 / **3** (not converged) / 2 |
| `centrality` | normalised degree centrality | 0 / **3** / 2 |
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
* **Two implementations must agree.** `compare` is the guarantee: Dijkstra and Bellman-Ford are
  independent code paths, and the command reports `identical`, the node count compared, both reachable
  counts, and every differing node — exiting **3** when they disagree.
* **Incremental labelling is honest.** `IncrementalComponents` merges components on insertion in
  near-constant time; a **removal** cannot be undone by union-find, so the structure marks itself stale,
  names the affected nodes, and makes `labels()` fail until `recompute()` is called instead of returning
  a quietly wrong answer.

## Layout

```
graphtk/graph.py        graph, edges, CSR, degrees, diagnostics
graphtk/algorithms.py   BFS/DFS, Dijkstra, Bellman-Ford, components, topological sort, PageRank, centrality
graphtk/incremental.py  union-find components with staleness tracking
graphtk/cli.py          eleven subcommands and the exit-code contract
tests/                  structure, traversal, shortest paths, ordering, ranking, incremental behaviour
```
