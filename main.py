#!/usr/bin/env python3

import argparse
import networkx as nx
import sys
import matplotlib.pyplot as plt
import re # Import regular expressions
import subprocess # To potentially call external demanglers if needed

# Attempt to import cxxfilt, provide instructions if missing
try:
    import cxxfilt
except ImportError:
    print("Error: The 'cxxfilt' library is required for demangling.")
    print("Please install it using: pip install cxxfilt")
    sys.exit(1)

# --- Demangling Function ---

def demangle_name(mangled_name):
    """
    Attempts to demangle a C++/Rust mangled name.

    Args:
        mangled_name (str): The mangled function name (e.g., starting with '@').

    Returns:
        str: The demangled name, or the original name if demangling fails or is not applicable.
    """
    if not mangled_name:
        return ""
    # Remove the leading '@' for demangling
    name_to_demangle = mangled_name.lstrip('@')
    try:
        # Use cxxfilt to demangle
        demangled = cxxfilt.demangle(name_to_demangle)
        # Basic simplification can happen here, but primary filtering uses the raw demangled name
        demangled = re.sub(r'::h[0-9a-f]{16}E$', '', demangled) # Remove trailing hash for non-generic functions
        return demangled
    except Exception:
        # Fallback to original name (without '@') if demangling fails
        return name_to_demangle

# --- LLVM IR Parsing Functions ---

def parse_llvm_ir(llvm_code):
    """
    Parses LLVM IR code to extract function definitions and calls.

    Args:
        llvm_code (str): The LLVM IR code as a string.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary mapping mangled function names to their definition lines.
            - list: A list of tuples representing calls (mangled_caller_name, mangled_callee_name).
    """
    functions = {}
    calls = []
    current_function = None

    # Regex to find function definitions (captures mangled function name)
    func_def_regex = re.compile(r"^\s*define\s+.*\s+(@[^(\s]+)\s*\(")

    # Regex to find call or invoke instructions (captures mangled function name)
    call_regex = re.compile(r"^\s*(?:tail\s+|musttail\s+|notail\s+)?(?:call|invoke)\s+.*\s+(@[^(\s]+)\s*\(")

    lines = llvm_code.splitlines()
    for line in lines:
        line = line.strip()

        # Check for function definition
        func_match = func_def_regex.match(line)
        if func_match:
            current_function = func_match.group(1)
            functions[current_function] = line
            continue

        # If inside a function, check for calls
        if current_function:
            call_match = call_regex.search(line)
            if call_match:
                callee_name = call_match.group(1)
                calls.append((current_function, callee_name))

            # Check if the current block ends (end of function)
            if line == "}":
                current_function = None

    return functions, calls

def build_call_graph_from_ll(llvm_code):
    """
    Builds a NetworkX DiGraph from LLVM IR code, creates labels, and filters
    the graph to show primarily user-defined functions based on naming conventions.

    Args:
        llvm_code (str): The LLVM IR code as a string.

    Returns:
        tuple: A tuple containing:
          - nx.DiGraph: The *filtered* call graph.
          - dict: A dictionary mapping mangled node IDs in the filtered graph
                  to their simplified demangled labels for display.
    """
    functions, calls = parse_llvm_ir(llvm_code)
    full_graph = nx.DiGraph()
    full_labels = {} # Dictionary for all labels {mangled_name: raw_demangled_name}

    # --- Step 1: Build the full graph and labels ---
    all_mangled_names = set(functions.keys())
    all_mangled_names.update(caller for caller, _ in calls)
    all_mangled_names.update(callee for _, callee in calls)

    for mangled_name in all_mangled_names:
        if not full_graph.has_node(mangled_name):
            full_graph.add_node(mangled_name)
            # Store the raw demangled name first for filtering
            full_labels[mangled_name] = demangle_name(mangled_name)

    for caller, callee in calls:
        if full_graph.has_node(caller) and full_graph.has_node(callee):
            full_graph.add_edge(caller, callee)

    # --- Step 2: Filter the graph ---
    filtered_graph = nx.DiGraph()
    filtered_display_labels = {} # Labels for the nodes kept in the filtered graph

    print("\nFiltering nodes (keeping non-stdlib, non-closure, non-generic, demangled names):")
    kept_nodes = 0
    filtered_out_nodes = 0
    for node, raw_label in full_labels.items():
        # Determine if the node should be kept based on the raw demangled label
        keep_node = True
        if raw_label.startswith('_ZN'): # Filter out if demangling failed
            keep_node = False
        elif raw_label.startswith('std::'): # Filter out standard library
            keep_node = False
        elif raw_label.startswith('core::'): # Filter out core library
            keep_node = False
        elif raw_label.startswith('alloc::'): # Filter out alloc library
             keep_node = False
        elif '{{closure}}' in raw_label: # Filter out closures
            keep_node = False
        elif '<' in raw_label or '>' in raw_label: # Filter out generics/impls
            keep_node = False
        # Add any other specific patterns to filter out here if needed
        # elif 'some_other_pattern' in raw_label:
        #     keep_node = False

        # If the node is kept, add it to the filtered graph and prepare its display label
        if keep_node:
            filtered_graph.add_node(node)
            # Simplify the label for display (remove common prefixes, etc.)
            simplified_label = raw_label # Start with the kept raw label
            simplified_label = re.sub(r'<.* as .*>::', '', simplified_label) # Simplify trait paths further if any slipped through
            # Optionally remove crate prefixes if they are repetitive, or keep them for clarity
            # Example: simplified_label = simplified_label.replace("my_crate_name::", "")
            filtered_display_labels[node] = simplified_label
            # print(f"  Keeping: {raw_label} -> {simplified_label} (Node: {node})")
            kept_nodes += 1
        else:
            # print(f"  Filtering out: {raw_label} (Node: {node})")
            filtered_out_nodes +=1
            pass # Don't add the node to the filtered graph

    print(f"Kept {kept_nodes} nodes, filtered out {filtered_out_nodes} nodes.")

    # --- Step 3: Add edges to the filtered graph ---
    #    This step ensures connectivity is maintained *between* the kept nodes,
    #    even if intermediate filtered-out nodes were involved in the original path.
    print("Adding edges to filtered graph (connecting kept nodes)...")
    added_edges = 0
    # Use transitive closure or path finding on the *full* graph to connect kept nodes
    kept_node_set = set(filtered_graph.nodes())
    for u in kept_node_set:
        # Find all nodes reachable from u in the full graph
        reachable_in_full = nx.descendants(full_graph, u)
        # Find which of those reachable nodes are also in our kept set
        kept_descendants = reachable_in_full.intersection(kept_node_set)
        # For each kept descendant, check if there's a *direct* edge in the full graph
        # If not, it implies the path went through filtered nodes.
        # We add a direct edge in the filtered graph to represent this reachability.
        # (Alternative: Find shortest path in full graph and add edge if path exists)

        # Simpler approach: Add direct edges from full graph if both ends are kept.
        # This is less accurate for showing indirect calls via filtered nodes,
        # but much simpler to implement and often sufficient.
        for v in full_graph.successors(u):
             if v in kept_node_set: # If successor is also a kept node
                 if not filtered_graph.has_edge(u,v): # Avoid duplicates
                     filtered_graph.add_edge(u, v)
                     added_edges += 1


    # To properly handle indirect calls (A calls B calls C, but B is filtered out -> show A calls C):
    # This requires more complex path analysis on the full graph.
    # For now, we stick to the simpler direct edge transfer. If indirect calls
    # need visualization, a more sophisticated edge creation step is required.

    print(f"Added {added_edges} direct edges between kept nodes.")


    # --- Step 4: Handle disconnected components (optional) ---
    # Remove nodes that have no connections within the filtered graph
    isolated_nodes = list(nx.isolates(filtered_graph))
    if isolated_nodes:
        print(f"Removing {len(isolated_nodes)} isolated nodes after filtering and edge addition.")
        filtered_graph.remove_nodes_from(isolated_nodes)
        for node in isolated_nodes:
             # Ensure label is removed if node is removed
            if node in filtered_display_labels:
                del filtered_display_labels[node]


    return filtered_graph, filtered_display_labels # Return filtered graph and its specific labels

# --- Graph Analysis and Visualization ---

def detect_cycles(graph):
    """
    Detects cycles (potential recursion) in the call graph.

    Args:
        graph (nx.DiGraph): The call graph.

    Returns:
        tuple: A tuple containing:
            - set: A set of mangled node names involved in any cycle.
            - set: A set of edges (mangled_u, mangled_v) involved in any cycle.
    """
    cyclic_nodes = set()
    cyclic_edges = set()
    if not graph.nodes: # Skip if graph is empty after filtering
        return cyclic_nodes, cyclic_edges
    try:
        # Find all simple cycles in the graph
        cycles = list(nx.simple_cycles(graph))
        for cycle in cycles:
            cyclic_nodes.update(cycle)
            # Add edges for the cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)] # Handle wrap-around
                if graph.has_edge(u, v):
                    cyclic_edges.add((u, v))
    except Exception as e:
        print(f"Error during cycle detection: {e}", file=sys.stderr)
        pass # Continue without cycle info if detection fails
    return cyclic_nodes, cyclic_edges


def draw_and_export_graph(graph, labels, svg_path, cyclic_nodes=None, cyclic_edges=None):
    """
    Draws the filtered graph using Matplotlib and exports it to SVG, highlighting cycles
    and using demangled labels, with spacing adjustments.

    Args:
        graph (nx.DiGraph): The *filtered* graph to draw.
        labels (dict): Mapping from mangled node names in the filtered graph to display labels.
        svg_path (str): The path to save the SVG file.
        cyclic_nodes (set, optional): Set of mangled node names involved in cycles. Defaults to None.
        cyclic_edges (set, optional): Set of edges (mangled_u, mangled_v) involved in cycles. Defaults to None.
    """
    if not graph.nodes:
        print("Filtered graph is empty, cannot draw.", file=sys.stderr)
        return

    # --- Adjustments for better spacing ---
    num_nodes = len(graph.nodes)
    # Increase figure size based on number of nodes (heuristic)
    fig_width = max(16, num_nodes * 0.8) # Adjusted scaling for potentially fewer nodes
    fig_height = max(13, num_nodes * 0.6) # Adjusted scaling
    plt.figure(figsize=(fig_width, fig_height))

    # Adjust k based on the number of nodes in the *filtered* graph
    k_value = 3.0 / (num_nodes**0.5) if num_nodes > 1 else 3.0 # Slightly reduced k base vs last time
    k_value = max(k_value, 0.3) # Ensure k doesn't get too small

    iterations = 100

    # Node sizes
    base_node_size = 1500 # Can increase size slightly as there are fewer nodes
    cyclic_node_size = 1700

    # Keep font size small
    font_size = 8 # Slightly larger font might be okay now
    # --- End Adjustments ---

    try:
        print(f"Calculating spring layout for filtered graph (k={k_value:.2f}, iterations={iterations})...")
        pos = nx.spring_layout(graph, seed=42, k=k_value, iterations=iterations)
        print("Layout calculation complete.")
    except Exception as layout_error:
        print(f"Warning: Spring layout failed ({layout_error}). Falling back to random layout.", file=sys.stderr)
        try:
            pos = nx.random_layout(graph, seed=42)
        except Exception as random_layout_error:
            print(f"Error: Random layout also failed ({random_layout_error}). Cannot generate positions.", file=sys.stderr)
            plt.close()
            return

    # Default node and edge colors/styles
    node_colors = ["#ccccff"] * len(graph.nodes)
    edge_colors = ["#aaaaaa"] * len(graph.edges)
    edge_widths = [1.0] * len(graph.edges) # Slightly thicker default edges
    node_sizes = [base_node_size] * len(graph.nodes)
    node_edge_colors = ["#333333"] * len(graph.nodes)

    node_list = list(graph.nodes) # List of mangled names in filtered graph
    edge_list = list(graph.edges) # List of edges in filtered graph

    # Highlight cyclic nodes and edges if found
    if cyclic_nodes:
        for i, node in enumerate(node_list):
            if node in cyclic_nodes:
                node_colors[i] = "#ffcccc"
                node_sizes[i] = cyclic_node_size
                node_edge_colors[i] = "#aa0000"

    if cyclic_edges:
        for i, edge in enumerate(edge_list):
            if edge in cyclic_edges:
                edge_colors[i] = "red"
                edge_widths[i] = 2.0

    try:
        print("Drawing filtered graph elements...")
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=edge_list,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.1',
            alpha=0.7 # Slightly less transparent edges
        )
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=node_list,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors=node_edge_colors
        )
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=font_size, font_family="monospace")

        file_base_name = svg_path.replace('.svg', '.ll')
        plt.title(f"Filtered LLVM IR Call Graph (User Functions) - Cycles Highlighted\nFile: {file_base_name}", fontsize=12) # Updated title
        plt.axis("off")
        plt.tight_layout()
        print(f"Saving filtered graph to {svg_path}...")
        plt.savefig(svg_path, format="svg", bbox_inches='tight', dpi=150)
        print(f"\nFiltered call graph exported as SVG: {svg_path}")

    except Exception as draw_error:
        print(f"Error during graph drawing or saving: {draw_error}", file=sys.stderr)
    finally:
        plt.close()


def main():
    parser_arg = argparse.ArgumentParser(
        description="Analyze LLVM IR code (.ll), filter graph to user functions, detect cycles, demangle names, and export as SVG."
    )
    parser_arg.add_argument("llvm_file", help="Path to the LLVM IR source file (.ll).")
    parser_arg.add_argument(
        "--svg", dest="svg_file", default=None,
        help="Path to output SVG file (default: <llvm_file>.filtered.svg)"
    )
    args = parser_arg.parse_args()

    try:
        with open(args.llvm_file, "r", encoding="utf-8") as f:
            llvm_code = f.read()
    except FileNotFoundError:
        print(f"Error: File not found '{args.llvm_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{args.llvm_file}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing LLVM IR file: {args.llvm_file}...")
    try:
        # Build and filter graph, get filtered labels
        call_graph, display_labels = build_call_graph_from_ll(llvm_code) # Now returns filtered graph/labels
    except Exception as e:
        print(f"Error building or filtering call graph: {e}", file=sys.stderr)
        sys.exit(1)


    if not call_graph.nodes:
        print("\nNo functions matching the filter criteria were found in the provided LLVM IR code.")
        sys.exit(0)
    else:
         print(f"\nFiltered graph contains {len(call_graph.nodes)} functions and {len(call_graph.edges)} calls.")


    # Cycle detection on the filtered graph
    print("Detecting cycles in filtered graph...")
    cyclic_nodes, cyclic_edges = detect_cycles(call_graph) # Use filtered graph

    if cyclic_nodes:
        sorted_mangled_cyclic_nodes = sorted(list(cyclic_nodes))
        print(f"\nDetected {len(sorted_mangled_cyclic_nodes)} function(s) involved in cycles within the filtered graph:")
        for mangled_node in sorted_mangled_cyclic_nodes:
            display_name = display_labels.get(mangled_node, mangled_node)
            print(f"  - {display_name}  ({mangled_node})")

    else:
        print("\nNo cycles detected in the filtered call graph.")


    svg_path = args.svg_file
    if not svg_path:
        base_name = args.llvm_file
        if base_name.lower().endswith(".ll"):
            base_name = base_name[:-3]
        svg_path = base_name + ".filtered.svg" # Changed default suffix

    print(f"\nGenerating filtered graph visualization...")
    # Pass the filtered graph and labels to the drawing function
    draw_and_export_graph(call_graph, display_labels, svg_path, cyclic_nodes=cyclic_nodes, cyclic_edges=cyclic_edges)


if __name__ == "__main__":
    main()
