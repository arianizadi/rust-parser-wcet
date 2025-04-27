#!/usr/bin/env python3

import argparse
import networkx as nx
from tree_sitter import Language, Parser
import tree_sitter_rust
import sys

# --- Tree-sitter Setup (Use tree_sitter_rust only) ---

RUST_LANGUAGE = None
parser = None

try:
    RUST_LANGUAGE = Language(tree_sitter_rust.language())
    print("RUST_LANGUAGE.version:", RUST_LANGUAGE.version)
    # Check for compatible version (must be between 13 and 14 inclusive)
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

    has_errors = False
    error_nodes = []
    queue = [root_node]
    processed_node_ids = set()

    while queue:
        node = queue.pop(0)
        if not node or node.id in processed_node_ids:
            continue
        processed_node_ids.add(node.id)

        is_error_node = node.type == "ERROR" or node.is_missing or node.has_error

        if is_error_node:
            has_errors = True
            error_nodes.append(node)
            continue

        queue.extend(node.children)

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


def main():
    parser_arg = argparse.ArgumentParser(
        description="Analyze Rust code to find the longest function call path between defined functions (within the file)."
    )
    parser_arg.add_argument("rust_file", help="Path to the Rust source file.")
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
        sys.exit(1)

    if longest_path:
        path_len_nodes = len(longest_path)
        print(
            f"\nLongest call path found (length {path_len_nodes} functions / {max(0, path_len_nodes - 1)} calls):"
        )
        path_str = [str(node) for node in longest_path]
        print(" -> ".join(path_str))
    else:
        print(
            "\nNo call paths found between the defined functions (graph might be disconnected or contain only single nodes)."
        )


if __name__ == "__main__":
    if RUST_LANGUAGE is None or parser is None:
        sys.exit(1)
    main()
