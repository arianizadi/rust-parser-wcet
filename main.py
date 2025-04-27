#!/usr/bin/env python3

import argparse
import networkx as nx
from tree_sitter import Language, Parser
import tree_sitter_rust
import sys
import matplotlib.pyplot as plt

RUST_LANGUAGE = None
parser = None

try:
    RUST_LANGUAGE = Language(tree_sitter_rust.language())
    if not (13 <= RUST_LANGUAGE.version <= 14):
        raise RuntimeError(
            f"Incompatible Language version {RUST_LANGUAGE.version}. Must be between 13 and 14"
        )
    parser = Parser(RUST_LANGUAGE)
except Exception as e:
    raise RuntimeError(f"Error loading Rust grammar for tree-sitter: {e}")


def find_nodes(node, node_type, results):
    if not node:
        return
    if node.type == node_type:
        results.append(node)
    for child in node.children:
        find_nodes(child, node_type, results)


def get_node_text(node, source_bytes):
    if not node:
        return ""
    start = max(0, node.start_byte)
    end = min(len(source_bytes), node.end_byte)
    if start >= end:
        return ""
    try:
        return source_bytes[start:end].decode("utf8", errors="replace")
    except Exception:
        return ""


def build_call_graph(source_code):
    source_bytes = bytes(source_code, "utf8")
    try:
        tree = parser.parse(source_bytes)
        root_node = tree.root_node
    except Exception as parse_error:
        raise RuntimeError(f"Error during tree-sitter parsing: {parse_error}")

    functions = {}
    calls = []

    function_nodes = []
    find_nodes(root_node, "function_item", function_nodes)

    for func_node in function_nodes:
        if not func_node or func_node.type == "ERROR" or func_node.is_missing:
            continue

        name_node = func_node.child_by_field_name("name")
        if name_node:
            if name_node.type != "ERROR" and not name_node.is_missing:
                func_name = get_node_text(name_node, source_bytes)
                if func_name:
                    functions[func_name] = func_node

    for caller_name, func_node in functions.items():
        body_node = func_node.child_by_field_name("body")
        if body_node and body_node.type != "ERROR" and not body_node.is_missing:
            local_call_nodes = []
            queue = [body_node]
            processed_nodes = set()
            while queue:
                current_node = queue.pop(0)
                if not current_node or current_node.id in processed_nodes:
                    continue
                processed_nodes.add(current_node.id)

                if (
                    current_node.type == "ERROR"
                    or current_node.is_missing
                    or current_node.has_error
                ):
                    continue

                if current_node.type == "call_expression":
                    local_call_nodes.append(current_node)

                queue.extend(current_node.children)

            for call_node in local_call_nodes:
                func_identifier_node = call_node.child_by_field_name("function")
                callee_name = None

                if (
                    func_identifier_node
                    and func_identifier_node.type != "ERROR"
                    and not func_identifier_node.is_missing
                ):
                    node_type = func_identifier_node.type
                    try:
                        if node_type == "identifier":
                            callee_name = get_node_text(
                                func_identifier_node, source_bytes
                            )
                        elif node_type == "scoped_identifier":
                            name_node = func_identifier_node.child_by_field_name("name")
                            path_node = func_identifier_node.child_by_field_name("path")
                            if name_node and name_node.type == "identifier":
                                callee_name = get_node_text(name_node, source_bytes)
                            elif (
                                path_node
                                and path_node.last_named_child
                                and path_node.last_named_child.type == "identifier"
                            ):
                                callee_name = get_node_text(
                                    path_node.last_named_child, source_bytes
                                )
                        elif node_type == "field_expression":
                            field_id_node = func_identifier_node.child_by_field_name(
                                "field"
                            )
                            if field_id_node and field_id_node.type == "identifier":
                                callee_name = get_node_text(field_id_node, source_bytes)
                    except Exception:
                        pass

                if callee_name and callee_name in functions:
                    calls.append((caller_name, callee_name))

    graph = nx.DiGraph()
    for func_name in functions:
        graph.add_node(func_name)
    for caller, callee in calls:
        if graph.has_node(caller) and graph.has_node(callee):
            graph.add_edge(caller, callee)

    return graph


def find_longest_path_in_graph(graph):
    if not graph or not graph.nodes:
        return []

    if not nx.is_directed_acyclic_graph(graph):
        return None

    try:
        longest_path_nodes = nx.dag_longest_path(graph, weight=None)
        return longest_path_nodes
    except nx.NetworkXNotImplemented:
        try:
            sorted_nodes = list(nx.topological_sort(graph))
            dist = {node: 1 for node in sorted_nodes}
            path_pred = {node: None for node in sorted_nodes}

            for u in sorted_nodes:
                for v, _ in graph.in_edges(u):
                    if dist[v] + 1 > dist[u]:
                        dist[u] = dist[v] + 1
                        path_pred[u] = v

            if not dist:
                return []
            end_node = max(dist, key=dist.get)
            max_len_nodes = dist[end_node]

            longest_path_nodes = []
            if max_len_nodes > 0:
                path = [end_node]
                curr = end_node
                while path_pred[curr] is not None:
                    path.append(path_pred[curr])
                    curr = path_pred[curr]
                longest_path_nodes = path[::-1]
            elif graph.nodes:
                longest_path_nodes = [end_node]

            return longest_path_nodes

        except Exception:
            return None

    except Exception:
        return None


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.3, vert_loc=0, xcenter=0.5, 
                  pos=None, parent=None):
    if pos is None:
        pos = {}
    if root is None:
        roots = [n for n, d in G.in_degree() if d == 0]
        if len(roots) > 0:
            root = roots[0]
        else:
            root = list(G.nodes)[0]
    children = list(G.successors(root))
    if not isinstance(G, nx.DiGraph):
        raise TypeError("hierarchy_pos requires a DiGraph")
    if len(children) == 0:
        pos[root] = (xcenter, vert_loc)
    else:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = hierarchy_pos(G, root=child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root)
        pos[root] = (xcenter, vert_loc)
    return pos


def draw_and_export_graph(graph, svg_path, highlight_path=None):
    plt.figure(figsize=(8, 6))

    root = None
    if "main" in graph.nodes:
        root = "main"
    else:
        roots = [n for n, d in graph.in_degree() if d == 0]
        if roots:
            root = roots[0]
        elif len(graph.nodes) > 0:
            root = list(graph.nodes)[0]
        else:
            root = None

    if root is not None and nx.is_directed_acyclic_graph(graph):
        pos = hierarchy_pos(graph, root=root, width=1.0, vert_gap=0.25, vert_loc=0, xcenter=0.5)
    else:
        pos = nx.spring_layout(graph, seed=42) if len(graph.nodes) > 1 else nx.shell_layout(graph)

    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle='-|>', connectionstyle='arc3,rad=0.1', alpha=0.5)
    nx.draw_networkx_nodes(graph, pos, node_color="#ccccff", node_size=1200, edgecolors="#333333")
    nx.draw_networkx_labels(graph, pos, font_size=11, font_family="monospace")

    if highlight_path and len(highlight_path) > 1:
        path_edges = list(zip(highlight_path, highlight_path[1:]))
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=path_edges,
            width=3,
            edge_color="red",
            arrows=True,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.1'
        )
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=highlight_path,
            node_color="#ffcccc",
            node_size=1400,
            edgecolors="#aa0000"
        )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(svg_path, format="svg")
    plt.close()


def main():
    parser_arg = argparse.ArgumentParser(
        description="Analyze Rust code to find the function call graph and export it as SVG."
    )
    parser_arg.add_argument("rust_file", help="Path to the Rust source file.")
    parser_arg.add_argument(
        "--svg", dest="svg_file", default=None,
        help="Path to output SVG file (default: <rust_file>.svg)"
    )
    args = parser_arg.parse_args()

    try:
        with open(args.rust_file, "r", encoding="utf-8") as f:
            rust_code = f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading file '{args.rust_file}': {e}")

    call_graph = build_call_graph(rust_code)

    if not call_graph.nodes:
        print("\nNo functions were found or successfully parsed in the provided code.")
        sys.exit(0)

    longest_path = find_longest_path_in_graph(call_graph)

    if longest_path is None:
        print("The call graph contains cycles (recursion). Cannot highlight the longest path.")
        highlight_path = None
    else:
        highlight_path = longest_path

    svg_path = args.svg_file
    if not svg_path:
        svg_path = args.rust_file.rsplit(".", 1)[0] + ".svg"

    draw_and_export_graph(call_graph, svg_path, highlight_path=highlight_path)

    print(f"\nCall graph exported as SVG: {svg_path}")
    if highlight_path and len(highlight_path) > 1:
        path_len_nodes = len(highlight_path)
        print(
            f"Longest call path highlighted (length {path_len_nodes} functions / {max(0, path_len_nodes - 1)} calls):"
        )
        path_str = [str(node) for node in highlight_path]
        print(" -> ".join(path_str))
    elif highlight_path and len(highlight_path) == 1:
        print(f"Only one function in the longest path: {highlight_path[0]}")
    else:
        print("No longest path highlighted (graph may be cyclic or disconnected).")


if __name__ == "__main__":
    if RUST_LANGUAGE is None or parser is None:
        sys.exit(1)
    main()
