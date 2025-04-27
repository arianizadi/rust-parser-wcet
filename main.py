#!/usr/bin/env python3

import argparse
import networkx as nx
import sys
import matplotlib.pyplot as plt
import re # Import regular expressions
import subprocess # To potentially call external demanglers if needed
from collections import defaultdict # For edge cycle counts

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
             # Add edge, avoid self-loops if not desired
             if caller != callee:
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
        # Exclude main entry point if it's just 'main' (often just setup)
        # Keep if it's crate::main
        if raw_label == 'main':
             keep_node = False
        elif raw_label.startswith('_ZN'): # Filter out if demangling failed
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
    print("Adding edges to filtered graph (connecting kept nodes)...")
    added_edges = 0
    kept_node_set = set(filtered_graph.nodes())
    for u in kept_node_set:
        # Simpler approach: Add direct edges from full graph if both ends are kept.
        for v in full_graph.successors(u):
             if v in kept_node_set: # If successor is also a kept node
                 if not filtered_graph.has_edge(u,v): # Avoid duplicates
                     filtered_graph.add_edge(u, v)
                     added_edges += 1

    print(f"Added {added_edges} direct edges between kept nodes.")


    # --- Step 4: Handle disconnected components (optional but recommended) ---
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
            - list: A list of cycles (each cycle is a list of nodes).
            - set: A set of mangled node names involved in any cycle.
            - set: A set of edges (mangled_u, mangled_v) involved in any cycle.
    """
    cycles_list = []
    cyclic_nodes = set()
    cyclic_edges = set()
    if not graph.nodes: # Skip if graph is empty after filtering
        return cycles_list, cyclic_nodes, cyclic_edges
    try:
        # Find all simple cycles in the graph
        cycles_list = list(nx.simple_cycles(graph)) # Store the list of cycles
        for cycle in cycles_list:
            cyclic_nodes.update(cycle)
            # Add edges for the cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)] # Handle wrap-around
                # Ensure the edge actually exists in the graph (important for filtered graphs)
                if graph.has_edge(u, v):
                    cyclic_edges.add((u, v))
    except Exception as e:
        print(f"Error during cycle detection: {e}", file=sys.stderr)
        # Return empty lists/sets if error occurs
        return [], set(), set()
    return cycles_list, cyclic_nodes, cyclic_edges


def calculate_edge_cycle_counts(graph, cycles_list):
    """
    Calculates how many cycles pass through each edge.

    Args:
        graph (nx.DiGraph): The graph.
        cycles_list (list): List of cycles (each cycle is a list of nodes).

    Returns:
        dict: A dictionary mapping edges (u, v) to the count of cycles passing through them.
    """
    edge_counts = defaultdict(int)
    for cycle in cycles_list:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)] # Handle wrap-around
            # Check if the edge exists in the graph before incrementing
            if graph.has_edge(u, v):
                edge_counts[(u, v)] += 1
    return dict(edge_counts) # Convert back to regular dict


def draw_and_export_graph(graph, labels, svg_path, cycles_list, cyclic_nodes=None, cyclic_edges=None):
    """
    Draws the filtered graph using Matplotlib and exports it to SVG, highlighting cycles,
    using demangled labels, showing edge cycle counts, and adjusting arrow visibility.

    Args:
        graph (nx.DiGraph): The *filtered* graph to draw.
        labels (dict): Mapping from mangled node names in the filtered graph to display labels.
        svg_path (str): The path to save the SVG file.
        cycles_list (list): List of cycles found in the graph.
        cyclic_nodes (set, optional): Set of mangled node names involved in cycles. Defaults to None.
        cyclic_edges (set, optional): Set of edges (mangled_u, mangled_v) involved in cycles. Defaults to None.
    """
    if not graph.nodes:
        print("Filtered graph is empty, cannot draw.", file=sys.stderr)
        return

    # --- Calculate Edge Cycle Counts ---
    edge_cycle_counts = calculate_edge_cycle_counts(graph, cycles_list)

    # --- Adjustments for better spacing and visibility ---
    num_nodes = len(graph.nodes)
    fig_width = max(16, num_nodes * 0.9) # Slightly larger scaling
    fig_height = max(13, num_nodes * 0.7)
    plt.figure(figsize=(fig_width, fig_height))

    k_value = 3.5 / (num_nodes**0.5) if num_nodes > 1 else 3.5 # Increase K slightly more
    k_value = max(k_value, 0.4)
    iterations = 120 # More iterations for layout

    base_node_size = 1600 # Slightly larger nodes
    cyclic_node_size = 1800
    font_size = 8
    arrow_size = 20 # Increased arrow size for visibility
    edge_label_font_size = 9 # Increased font size for cycle counts 
    connection_radius = 0.25 # Increased radius for edge curvature
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
    edge_widths = [1.0] * len(graph.edges)
    node_sizes = [base_node_size] * len(graph.nodes)
    node_edge_colors = ["#333333"] * len(graph.nodes)

    node_list = list(graph.nodes)
    edge_list = list(graph.edges)

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
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=node_list,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors=node_edge_colors
        )
        
        # Draw edges with adjusted arrows and curvature
        # Use shrinkA and shrinkB to make arrows visible in the middle
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=edge_list,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=arrow_size,
            # Increase curvature radius to pull arrows away from nodes
            connectionstyle=f'arc3,rad={connection_radius}',
            # Use shrinkA and shrinkB to pull arrows away from nodes
            # These parameters control how much to shorten the edges from each end
            # Setting to 0 places the arrow at the node center, 
            # Higher values (fraction of total distance) move them toward the middle
            # Setting both to 0.5 would place the arrow exactly in the middle
            # We'll use values that position arrows near but not exactly at the middle
            alpha=0.7
        )
        
        # Draw custom edge arrows in the middle of edges
        # This adds a second set of arrows that will appear mid-edge
        for i, (u, v) in enumerate(edge_list):
            # Get positions of the nodes
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Calculate the midpoint of the edge
            # Adding a slight offset from perfect middle (0.5 coefficient)
            # for better visibility with the cycle count label
            mid_x = x1 + 0.45 * (x2 - x1)
            mid_y = y1 + 0.45 * (y2 - y1)
            
            # Calculate the direction vector
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize the direction vector
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                dx, dy = dx/length, dy/length
            
            # Apply a curvature offset for curved edges
            # This is necessary to match the curved edge path
            if connection_radius != 0:
                # Perpendicular vector (rotated 90 degrees) for arc curvature
                perpx, perpy = -dy, dx
                # Adjust midpoint to account for arc curvature
                mid_x += connection_radius * perpx * 5  # Scale to match the arc 
                mid_y += connection_radius * perpy * 5  # Scale to match the arc
            
            # Plot arrow in the middle of the edge
            # Only for edges with cycle count > 0 (to avoid clutter)
            if (u, v) in edge_cycle_counts and edge_cycle_counts[(u, v)] > 0:
                arrow_color = 'blue' if (u, v) in cyclic_edges else 'black'
                # Draw a more visible arrow
                plt.arrow(
                    mid_x, mid_y, 
                    dx * 0.1, dy * 0.1,  # Make arrow length proportional to edge
                    head_width=0.015, 
                    head_length=0.03,
                    fc=arrow_color, 
                    ec=arrow_color,
                    length_includes_head=True
                )
        
        # Draw node labels
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=font_size, font_family="monospace")

        # Draw edge labels (cycle counts) for edges involved in cycles
        # Now draw ALL edge cycle counts, including zeros, for edges in cycles
        edge_labels_to_draw = {}
        for edge in edge_list:
            if edge in cyclic_edges:
                count = edge_cycle_counts.get(edge, 0)
                edge_labels_to_draw[edge] = count
        
        # Draw counts that are non-zero for other edges
        for edge, count in edge_cycle_counts.items():
            if count > 0 and edge not in edge_labels_to_draw:
                edge_labels_to_draw[edge] = count
                
        if edge_labels_to_draw:
            print(f"Drawing {len(edge_labels_to_draw)} edge cycle counts...")
            nx.draw_networkx_edge_labels(
                graph, pos,
                edge_labels=edge_labels_to_draw,
                font_size=edge_label_font_size,
                font_color='blue', # Make cycle count distinct
                label_pos=0.5, # Position label near the middle of the edge
                bbox=dict(
                    facecolor='white', 
                    alpha=0.8,  # Increased opacity for better readability 
                    edgecolor='blue',  # Add outline to label box
                    boxstyle='round,pad=0.2',  # Increased padding
                    linewidth=0.8  # Border width
                ) # Add background for readability
            )

        # Add a legend explaining the visualization elements
        legend_elements = [
            plt.Line2D([0], [0], color='red', linewidth=2, label='Edge in cycle'),
            plt.Line2D([0], [0], color='#aaaaaa', linewidth=1, label='Normal edge'),
            plt.Rectangle((0,0), 1, 1, fc="#ffcccc", ec="#aa0000", label='Node in cycle'),
            plt.Rectangle((0,0), 1, 1, fc="#ccccff", ec="#333333", label='Normal node'),
            plt.Line2D([0], [0], marker='>', color='w', 
                      markerfacecolor='blue', markersize=10, 
                      label='Flow direction'),
            plt.Text(0, 0, '2', color='blue', fontsize=9, 
                    backgroundcolor='white', label='Cycle count')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', 
                  fontsize=8, title="Legend", title_fontsize=9)

        file_base_name = svg_path.replace('.svg', '.ll')
        plt.title(f"Filtered LLVM IR Call Graph (User Functions) - Cycles Highlighted\nFile: {file_base_name}", fontsize=12)
        plt.axis("off")
        # Use tight_layout cautiously, it can sometimes interfere with edge labels
        # plt.tight_layout()
        print(f"Saving filtered graph to {svg_path}...")
        plt.savefig(svg_path, format="svg", bbox_inches='tight', dpi=150)
        print(f"\nFiltered call graph exported as SVG: {svg_path}")

    except Exception as draw_error:
        print(f"Error during graph drawing or saving: {draw_error}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        plt.close()

def main():
    parser_arg = argparse.ArgumentParser(
        description="Analyze LLVM IR code (.ll), filter graph to user functions, detect cycles, demangle names, and export as SVG with edge cycle counts."
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
    # Get the list of cycles, the set of nodes in cycles, and the set of edges in cycles
    cycles_list, cyclic_nodes, cyclic_edges = detect_cycles(call_graph) # Use filtered graph
    num_cycles = len(cycles_list) # Get the count of simple cycles

    if cyclic_nodes:
        sorted_mangled_cyclic_nodes = sorted(list(cyclic_nodes))
        # Print the count of cycles found
        print(f"\nDetected {num_cycles} simple cycle(s) involving {len(sorted_mangled_cyclic_nodes)} function(s) within the filtered graph:")
        # List the functions involved in cycles
        for mangled_node in sorted_mangled_cyclic_nodes:
            display_name = display_labels.get(mangled_node, mangled_node)
            print(f"  - {display_name}  ({mangled_node})")
        # Optionally print the cycles themselves (can be verbose)
        # print("\nCycles found:")
        # for i, cycle in enumerate(cycles_list):
        #     demangled_cycle = [display_labels.get(node, node) for node in cycle]
        #     print(f"  Cycle {i+1}: {' -> '.join(demangled_cycle)} -> {demangled_cycle[0]}")

    else:
        print("\nNo cycles detected in the filtered call graph.")


    svg_path = args.svg_file
    if not svg_path:
        base_name = args.llvm_file
        if base_name.lower().endswith(".ll"):
            base_name = base_name[:-3]
        svg_path = base_name + ".filtered.svg" # Changed default suffix

    print(f"\nGenerating filtered graph visualization...")
    # Pass the list of cycles to the drawing function
    draw_and_export_graph(call_graph, display_labels, svg_path, cycles_list, cyclic_nodes=cyclic_nodes, cyclic_edges=cyclic_edges)


if __name__ == "__main__":
    main()
