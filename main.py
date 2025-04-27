#!/usr/bin/env python3

import argparse
import networkx as nx
import sys
import matplotlib.pyplot as plt
import re
import subprocess
from collections import defaultdict

# Attempt to import cxxfilt, provide instructions if missing
try:
    import cxxfilt
except ImportError:
    print("Error: The 'cxxfilt' library is required for demangling.")
    print("Please install it using: pip install cxxfilt")
    sys.exit(1)

def demangle_name(mangled_name):
    """Demangles a C++ mangled name using cxxfilt."""
    if not mangled_name:
        return ""
    # Strip leading '@' which is common in LLVM IR identifiers
    name_to_demangle = mangled_name.lstrip('@')
    try:
        # Demangle the name
        demangled = cxxfilt.demangle(name_to_demangle)
        # Remove potential Rust hash suffixes like ::h<16 hex digits>E
        demangled = re.sub(r'::h[0-9a-f]{16}E$', '', demangled)
        return demangled
    except Exception:
        # If demangling fails, return the original name (without '@')
        return name_to_demangle

def parse_llvm_ir(llvm_code):
    """Parses LLVM IR code to find function definitions and call sites."""
    functions = {}  # Dictionary to store function definitions (mangled_name: definition_line)
    calls = []      # List to store calls (caller_mangled_name, callee_mangled_name)
    current_function = None # Track the function currently being parsed

    # Regex to find function definitions (e.g., "define ... @function_name(...) ... {")
    # Captures the function name (group 1)
    func_def_regex = re.compile(r"^\s*define\s+.*\s+(@[^(\s]+)\s*\(")

    # Regex to find call or invoke instructions
    # Captures the called function name (group 1)
    # Handles optional prefixes like tail, musttail, notail
    call_regex = re.compile(r"^\s*(?:tail\s+|musttail\s+|notail\s+)?(?:call|invoke)\s+.*\s+(@[^(\s]+)\s*\(")

    lines = llvm_code.splitlines()
    for line in lines:
        line = line.strip() # Remove leading/trailing whitespace

        # Check for function definition
        func_match = func_def_regex.match(line)
        if func_match:
            current_function = func_match.group(1) # Get the mangled name
            functions[current_function] = line    # Store the definition line
            continue # Move to the next line

        # If inside a function definition
        if current_function:
            # Check for call/invoke instructions
            call_match = call_regex.search(line)
            if call_match:
                callee_name = call_match.group(1) # Get the called function's mangled name
                calls.append((current_function, callee_name)) # Record the call

            # Check for the end of the function definition
            if line == "}":
                current_function = None # Reset current function tracker

    return functions, calls

def build_call_graph_from_ll(llvm_code):
    """Builds a filtered call graph from LLVM IR code."""
    functions, calls = parse_llvm_ir(llvm_code) # Get functions and calls

    # --- Build the Full Graph ---
    full_graph = nx.DiGraph() # Create an empty directed graph
    full_labels = {}          # Dictionary for demangled labels {mangled_name: demangled_name}

    # Collect all unique function names (defined or called)
    all_mangled_names = set(functions.keys())
    all_mangled_names.update(caller for caller, _ in calls)
    all_mangled_names.update(callee for _, callee in calls)

    # Add nodes to the full graph and store demangled labels
    for mangled_name in all_mangled_names:
        if not full_graph.has_node(mangled_name):
            full_graph.add_node(mangled_name)
            full_labels[mangled_name] = demangle_name(mangled_name) # Demangle and store

    # Add edges to the full graph based on calls
    for caller, callee in calls:
        # Ensure both caller and callee nodes exist (should always be true here)
        if full_graph.has_node(caller) and full_graph.has_node(callee):
            # Avoid self-loops in the visualization (can be adjusted if needed)
            if caller != callee:
                full_graph.add_edge(caller, callee)

    # --- Filter the Graph ---
    filtered_graph = nx.DiGraph()      # Create the graph for filtered nodes
    filtered_display_labels = {}       # Labels for the filtered graph

    print("\nFiltering nodes (keeping non-stdlib, non-closure, non-generic, demangled names):")
    kept_nodes = 0
    filtered_out_nodes = 0

    # Iterate through nodes in the full graph
    for node, raw_label in full_labels.items():
        keep_node = True # Assume we keep the node initially

        # --- Filtering Criteria ---
        # Exclude specific patterns or namespaces
        if raw_label == 'main':             # Exclude 'main' itself if desired
             keep_node = False
        elif raw_label.startswith('_ZN'):   # Exclude mangled names starting with _ZN (often internal/compiler-generated)
             keep_node = False
        elif raw_label.startswith('std::'): # Exclude C++ standard library
            keep_node = False
        elif raw_label.startswith('core::'): # Exclude Rust core library
            keep_node = False
        elif raw_label.startswith('alloc::'): # Exclude Rust alloc library
            keep_node = False
        elif '{{closure}}' in raw_label:    # Exclude closures
            keep_node = False
        # Exclude generic functions (containing '<' or '>') - adjust if needed
        elif '<' in raw_label or '>' in raw_label:
            keep_node = False
        # --- End Filtering Criteria ---

        if keep_node:
            # Add the node to the filtered graph
            filtered_graph.add_node(node)
            # Simplify label further (e.g., remove trait implementation details)
            simplified_label = re.sub(r'<.* as .*>::', '', raw_label)
            filtered_display_labels[node] = simplified_label # Store the display label
            kept_nodes += 1
        else:
            filtered_out_nodes += 1

    print(f"Kept {kept_nodes} nodes, filtered out {filtered_out_nodes} nodes.")

    # --- Add Edges to Filtered Graph ---
    print("Adding edges to filtered graph (connecting kept nodes)...")
    added_edges = 0
    kept_node_set = set(filtered_graph.nodes()) # Efficient lookup for kept nodes

    # Iterate through the kept nodes
    for u in kept_node_set:
        # Find successors (nodes called by u) in the *full* graph
        for v in full_graph.successors(u):
            # If the successor is also a kept node
            if v in kept_node_set:
                # Add the edge to the filtered graph if it doesn't exist
                if not filtered_graph.has_edge(u, v):
                    filtered_graph.add_edge(u, v)
                    added_edges += 1

    print(f"Added {added_edges} direct edges between kept nodes.")

    # --- Remove Isolated Nodes ---
    # Nodes that were kept but have no connections *within the filtered set*
    isolated_nodes = list(nx.isolates(filtered_graph))
    if isolated_nodes:
        print(f"Removing {len(isolated_nodes)} isolated nodes after filtering and edge addition.")
        filtered_graph.remove_nodes_from(isolated_nodes)
        # Remove labels for isolated nodes
        for node in isolated_nodes:
            if node in filtered_display_labels:
                del filtered_display_labels[node]

    return filtered_graph, filtered_display_labels

def detect_cycles(graph):
    """Detects simple cycles in a directed graph."""
    cycles_list = []    # List to store cycles found
    cyclic_nodes = set() # Set of nodes participating in any cycle
    cyclic_edges = set() # Set of edges participating in any cycle

    # Check if the graph has nodes before attempting cycle detection
    if not graph.nodes:
        return cycles_list, cyclic_nodes, cyclic_edges

    try:
        # Use networkx function to find all simple cycles (no repeated nodes except start/end)
        cycles_generator = nx.simple_cycles(graph)
        cycles_list = list(cycles_generator) # Convert generator to list

        # Populate sets of cyclic nodes and edges
        for cycle in cycles_list:
            cyclic_nodes.update(cycle) # Add all nodes in the cycle
            # Add edges forming the cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)] # Next node in cycle (wraps around)
                if graph.has_edge(u, v): # Ensure the edge actually exists
                    cyclic_edges.add((u, v))
    except Exception as e:
        # Handle potential errors during cycle detection (e.g., complex graph issues)
        print(f"Error during cycle detection: {e}", file=sys.stderr)
        return [], set(), set() # Return empty sets/list on error

    return cycles_list, cyclic_nodes, cyclic_edges

def calculate_edge_cycle_counts(graph, cycles_list):
    """Calculates how many simple cycles each edge belongs to."""
    edge_counts = defaultdict(int) # Default count is 0
    for cycle in cycles_list:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            # Check if the edge exists in the graph (important for correctness)
            if graph.has_edge(u, v):
                edge_counts[(u, v)] += 1 # Increment count for this edge
    return dict(edge_counts) # Convert back to a regular dict

def calculate_node_cycle_counts(graph, cycles_list):
    """Calculates how many simple cycles each node belongs to."""
    node_counts = defaultdict(int) # Default count is 0
    # Iterate through unique cycles (represented as tuples of nodes)
    unique_cycles = set(tuple(sorted(cycle)) for cycle in cycles_list) # Use sorted tuple to handle different starting points
    
    # We need the original cycles list to count participation correctly
    for cycle in cycles_list:
        # Iterate through unique nodes within *this specific* cycle
        for node in set(cycle):
             node_counts[node] += 1 # Increment count for the node
    return dict(node_counts) # Convert back to a regular dict


def estimate_wcet_cycles(function_name, llvm_code):
    """
    Estimate WCET (Worst-Case Execution Time) in cycles for a function
    by analyzing its LLVM IR instructions. This is a heuristic approach.
    """
    try:
        # --- Extract Function Body ---
        # Escape the function name for regex and handle potential '@' prefix
        escaped_name = re.escape(function_name.lstrip('@'))
        # Regex to find the specific function definition and capture its body
        # It looks for 'define ... @func_name(...) ... {' and captures everything until the matching '}'
        # Uses non-greedy matching (.*?) and DOTALL flag to handle multi-line bodies
        function_pattern = re.compile(
            r'define.*?@' + escaped_name + r'\s*\(.*?\)\s*.*?\{(.*?)\n\}',
            re.DOTALL | re.MULTILINE
        )
        match = function_pattern.search(llvm_code)
        if not match:
            # print(f"Warning: Could not find body for {function_name}", file=sys.stderr)
            return None # Function body not found

        function_body = match.group(1)

        # --- Instruction Counting with Estimated Cycle Costs ---
        # Initialize cycle count
        estimated_cycles = 0

        # Define approximate cycle costs (adjust based on target architecture assumptions)
        # These are *heuristics* and averages. Real costs vary wildly.
        costs = {
            'load': 5,          # Memory access (cache hit assumed, miss is much higher)
            'store': 5,         # Memory access
            'alloca': 2,        # Stack allocation
            'add': 1, 'sub': 1, # Integer arithmetic (simple)
            'mul': 3,           # Integer multiplication
            'sdiv': 15, 'udiv': 15, # Integer division (expensive)
            'srem': 15, 'urem': 15, # Integer remainder (expensive)
            'fadd': 4, 'fsub': 4, # Floating point add/sub
            'fmul': 5,          # Floating point multiply
            'fdiv': 20,         # Floating point division (very expensive)
            'frem': 20,         # Floating point remainder
            'icmp': 1, 'fcmp': 2,# Comparison (integer/float)
            'br': 2,            # Branch (average conditional/unconditional) - Penalize branches
            'switch': 5,        # Switch instruction (base cost)
            'call': 20,         # Function call overhead (significant)
            'invoke': 25,       # Call with exception handling (slightly more)
            'ret': 2,           # Return instruction
            'phi': 1,           # PHI node (data flow merge)
            'select': 1,        # Select instruction
            'getelementptr': 1, # Address calculation
            # Add other common instructions if needed
        }

        # Count instructions and accumulate costs
        instruction_count = 0
        lines = function_body.splitlines()
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith(';') or stripped_line.endswith(':'):
                continue # Skip empty lines, comments, labels

            # Extract the first word, which is usually the opcode
            match = re.match(r'\s*(?:%\w+\s*=\s*)?(\w+)', stripped_line)
            if match:
                opcode = match.group(1)
                # Accumulate cost if opcode is in our cost map
                if opcode in costs:
                    estimated_cycles += costs[opcode]
                    instruction_count += 1
                # Handle specific cases like branches
                #elif opcode == 'br' and 'label %' in stripped_line: # Unconditional branch
                #    estimated_cycles += 1 # Lower cost for unconditional
                #    instruction_count += 1

        # --- Complexity Factors ---
        # Basic Blocks: Add a small overhead per block transition
        basic_blocks = len(re.findall(r'^\s*\w+:', function_body, re.MULTILINE))
        estimated_cycles += basic_blocks * 1 # Add 1 cycle per block for potential jumps

        # --- Minimum Cost ---
        # Ensure a small minimum cost even for very simple functions
        # Use instruction count as a proxy for minimum complexity
        min_cost = 5 + instruction_count # Base cost + 1 per instruction found
        estimated_cycles = max(estimated_cycles, min_cost)

        # --- Loop Heuristic (Very Basic) ---
        # Count back-edges (branches to labels defined earlier in the *textual* order)
        # This is a *very rough* proxy for loops. Real loop analysis is complex.
        labels = {}
        back_edges = 0
        for i, line in enumerate(lines):
             label_match = re.match(r'^\s*(\w+):', line.strip())
             if label_match:
                 labels[label_match.group(1)] = i # Store label name and line number

        for i, line in enumerate(lines):
            branch_match = re.search(r'\bbr\s+label\s+%(\w+)', line)
            if branch_match:
                target_label = branch_match.group(1)
                if target_label in labels and labels[target_label] < i:
                    back_edges += 1 # Found a potential back-edge

        # Increase cost significantly if potential loops are detected
        if back_edges > 0:
             # Add a penalty proportional to the number of back-edges and estimated base cost
             loop_penalty = back_edges * (estimated_cycles * 0.5) # Arbitrary penalty factor
             estimated_cycles += int(loop_penalty)


        return estimated_cycles # Return the final estimated cycle count

    except Exception as e:
        # Print error details if estimation fails
        print(f"Error estimating cycles for {function_name}: {e}", file=sys.stderr)
        demangled_name = demangle_name(function_name)
        print(f"  - {demangled_name}: Unable to estimate cycles")
        return None # Return None on error


def draw_and_export_graph(graph, labels, svg_path, cycles_list, llvm_code, cyclic_nodes=None, cyclic_edges=None):
    """
    Draws the filtered graph using Matplotlib and exports it to SVG, highlighting cycles,
    using demangled labels, and showing estimated WCET cycles under each node.
    """
    if not graph.nodes:
        print("Filtered graph is empty, cannot draw.", file=sys.stderr)
        return

    # --- Calculate Edge and Node Cycle Counts ---
    edge_cycle_counts = calculate_edge_cycle_counts(graph, cycles_list)
    node_cycle_counts = calculate_node_cycle_counts(graph, cycles_list)

    # --- Calculate WCET estimates for each node ---
    wcet_estimates = {}
    print("\nEstimating WCET cycles for each function in the filtered graph...")
    total_estimated_cycles = 0
    estimated_nodes = 0
    for node in graph.nodes:
        wcet_cycles = estimate_wcet_cycles(node, llvm_code) # Use the updated function
        if wcet_cycles is not None: # Check if estimation was successful
            wcet_estimates[node] = wcet_cycles
            display_name = labels.get(node, node) # Get demangled name for printing
            print(f"  - {display_name}: ~{wcet_cycles} cycles")
            total_estimated_cycles += wcet_cycles
            estimated_nodes += 1
        else:
            display_name = labels.get(node, node)
            print(f"  - {display_name}: Estimation failed")

    if estimated_nodes > 0:
        avg_cycles = total_estimated_cycles / estimated_nodes
        print(f"\nAverage estimated WCET per function (in filtered graph): ~{avg_cycles:.1f} cycles")


    # --- Graph Drawing Setup ---
    num_nodes = len(graph.nodes)
    # Adjust figure size based on the number of nodes for better layout
    fig_width = max(18, num_nodes * 1.5) # Increased multiplier
    fig_height = max(15, num_nodes * 1.2) # Increased multiplier
    plt.figure(figsize=(fig_width, fig_height))

    # Spring layout parameters (adjust k for spacing, iterations for convergence)
    # Lower k spreads nodes out more.
    k_value = 5.0 / (num_nodes**0.5) if num_nodes > 1 else 5.0 # Slightly increased base
    k_value = max(k_value, 0.7) # Ensure minimum separation
    iterations = 250 # More iterations for potentially better layout

    # Node and edge styling parameters
    base_node_size = 2500 # Slightly larger base size
    cyclic_node_size = 3000 # Larger size for nodes in cycles
    font_size = 10 # Slightly smaller font for potentially long names
    arrow_size = 35 # Slightly smaller arrows
    connection_radius = 0.3 # Curve for edges

    # --- Calculate Layout ---
    try:
        print(f"\nCalculating spring layout for filtered graph (k={k_value:.2f}, iterations={iterations})...")
        pos = nx.spring_layout(graph, seed=42, k=k_value, iterations=iterations)
        print("Layout calculation complete.")
    except Exception as layout_error:
        print(f"Warning: Spring layout failed ({layout_error}). Falling back to random layout.", file=sys.stderr)
        try:
            pos = nx.random_layout(graph, seed=42) # Fallback layout
        except Exception as random_layout_error:
            print(f"Error: Random layout also failed ({random_layout_error}). Cannot generate positions.", file=sys.stderr)
            plt.close() # Close the plot figure
            return

    # --- Prepare Colors, Sizes, Widths ---
    node_list = list(graph.nodes)
    edge_list = list(graph.edges)

    # Default styles
    node_colors = ["#d0d0ff"] * len(node_list) # Lighter blue nodes
    edge_colors = ["#b0b0b0"] * len(edge_list) # Lighter gray edges
    edge_widths = [1.5] * len(edge_list)
    node_sizes = [base_node_size] * len(node_list)
    node_edge_colors = ["#444444"] * len(node_list) # Darker node borders

    # Apply styles for cyclic nodes and edges
    if cyclic_nodes:
        for i, node in enumerate(node_list):
            if node in cyclic_nodes:
                node_colors[i] = "#ffcccc" # Light red for cyclic nodes
                node_sizes[i] = cyclic_node_size
                node_edge_colors[i] = "#aa0000" # Dark red border

    if cyclic_edges:
        for i, edge in enumerate(edge_list):
            if edge in cyclic_edges:
                edge_colors[i] = "#ff6666" # Brighter red for cyclic edges
                edge_widths[i] = 3.0 # Thicker cyclic edges

    # --- Draw Graph Elements ---
    try:
        print("Drawing filtered graph elements...")

        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=node_list,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors=node_edge_colors, # Border color
            linewidths=2.0 # Border width
        )

        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=edge_list,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowstyle='-|>', # Arrow style
            arrowsize=arrow_size,
            connectionstyle=f'arc3,rad={connection_radius}', # Curved edges
            min_source_margin=25, # Margin around node for arrow start
            min_target_margin=25, # Margin around node for arrow end
            alpha=0.8 # Slight transparency
        )

        # Draw node labels (function names) inside nodes
        nx.draw_networkx_labels(
            graph, pos,
            labels=labels, # Use the demangled/simplified labels
            font_size=font_size,
            font_family="sans-serif", # Use a standard sans-serif font
            font_weight="bold",
            font_color="#111111" # Very dark gray/black
        )

        # Draw WCET cycle estimates *below* each node
        label_vertical_offset = -0.06 # Adjust as needed based on node size/font size
        for node in node_list:
            if node in pos: # Ensure position exists
                x, y = pos[node]
                # Compose the label text
                if node in wcet_estimates:
                    wcet = wcet_estimates[node]
                    # Format large numbers for readability
                    if wcet >= 10000:
                       label_text = f"~{wcet/1000:.1f}k cyc"
                    elif wcet >= 1000:
                       label_text = f"~{wcet/1000:.1f}k cyc" # Or just f"~{wcet} cyc"
                    else:
                       label_text = f"~{wcet} cyc"
                    label_color = "#00008B" # Dark Blue
                else:
                    label_text = "WCET: N/A"
                    label_color = "#888888" # Gray for unavailable

                # Draw the WCET label text below the node
                plt.text(
                    x, y + label_vertical_offset, # Position below the node center
                    label_text,
                    fontsize=font_size - 1, # Slightly smaller than node label
                    ha='center', # Horizontal alignment
                    va='top',    # Vertical alignment
                    color=label_color,
                    fontweight='normal',
                    # Optional: Add a subtle background for readability if needed
                    # bbox=dict(facecolor='white', alpha=0.7, pad=0.1, edgecolor='none')
                )

        # (Optional) Draw edge labels for cycle counts - can clutter the graph
        # edge_labels = {edge: edge_cycle_counts.get(edge, 0) for edge in edge_list if edge_cycle_counts.get(edge, 0) > 0}
        # if edge_labels:
        #     nx.draw_networkx_edge_labels(
        #         graph, pos,
        #         edge_labels=edge_labels,
        #         font_color='red',
        #         font_size=font_size - 2,
        #         label_pos=0.6 # Position along the edge
        #     )

        # --- Final Plot Adjustments ---
        file_base_name = svg_path.split('/')[-1].replace('.filtered.svg', '.ll') # Get base LLVM filename
        plt.title(f"Filtered LLVM IR Call Graph (User Functions) - Cycles Highlighted\nSource: {file_base_name}", fontsize=16, y=1.02) # Adjust title position
        plt.axis("off") # Hide axes
        plt.tight_layout(pad=1.0) # Adjust padding around the graph

        # --- Save the Figure ---
        print(f"Saving filtered graph to {svg_path}...")
        plt.savefig(svg_path, format="svg", bbox_inches='tight', dpi=150) # Use reasonable DPI
        print(f"\nFiltered call graph exported as SVG: {svg_path}")

    except Exception as draw_error:
        print(f"Error during graph drawing or saving: {draw_error}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        plt.close() # Ensure the plot figure is closed

def main():
    # --- Argument Parsing ---
    parser_arg = argparse.ArgumentParser(
        description="Analyze LLVM IR (.ll), filter call graph, detect cycles, estimate WCET per function, demangle names, and export as SVG."
    )
    parser_arg.add_argument("llvm_file", help="Path to the LLVM IR source file (.ll).")
    parser_arg.add_argument(
        "--svg", dest="svg_file", default=None,
        help="Path to output SVG file (default: <llvm_file_base>.filtered.svg)"
    )
    args = parser_arg.parse_args()

    # --- File Reading ---
    try:
        print(f"Reading LLVM IR file: {args.llvm_file}...")
        with open(args.llvm_file, "r", encoding="utf-8") as f:
            llvm_code = f.read()
        print("File reading complete.")
    except FileNotFoundError:
        print(f"Error: File not found '{args.llvm_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{args.llvm_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Graph Building and Filtering ---
    print(f"\nBuilding and filtering call graph from {args.llvm_file}...")
    try:
        call_graph, display_labels = build_call_graph_from_ll(llvm_code)
    except Exception as e:
        print(f"Error building or filtering call graph: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Check if filtering resulted in any nodes
    if not call_graph.nodes:
        print("\nNo functions matching the filter criteria were found. Exiting.")
        sys.exit(0)
    else:
        print(f"\nFiltered graph contains {len(call_graph.nodes)} functions and {len(call_graph.edges)} calls.")

    # --- Cycle Detection ---
    print("\nDetecting cycles in the filtered graph...")
    cycles_list, cyclic_nodes, cyclic_edges = detect_cycles(call_graph)
    num_cycles = len(cycles_list)

    if cyclic_nodes:
        # Sort nodes for consistent output
        sorted_mangled_cyclic_nodes = sorted(list(cyclic_nodes), key=lambda n: display_labels.get(n, n))
        print(f"\nDetected {num_cycles} simple cycle(s) involving {len(sorted_mangled_cyclic_nodes)} function(s):")
        for mangled_node in sorted_mangled_cyclic_nodes:
            display_name = display_labels.get(mangled_node, mangled_node) # Get the nice name
            # Optionally show cycle count per node
            node_cycle_count = calculate_node_cycle_counts(call_graph, cycles_list).get(mangled_node, 0)
            print(f"  - {display_name} (in {node_cycle_count} cycle(s))") # ({mangled_node})
    else:
        print("\nNo cycles detected in the filtered call graph.")

    # --- Determine Output SVG Path ---
    svg_path = args.svg_file
    if not svg_path:
        # Create default SVG name based on input LLVM file name
        base_name = args.llvm_file
        if base_name.lower().endswith(".ll"):
            base_name = base_name[:-3] # Remove .ll extension
        svg_path = base_name + ".filtered.svg" # Add .filtered.svg

    # --- Graph Drawing and Export ---
    print(f"\nGenerating filtered graph visualization (output: {svg_path})...")
    # Pass all necessary data to the drawing function
    draw_and_export_graph(
        call_graph,
        display_labels,
        svg_path,
        cycles_list,
        llvm_code, # Pass the full LLVM code for WCET estimation
        cyclic_nodes=cyclic_nodes,
        cyclic_edges=cyclic_edges
    )

if __name__ == "__main__":
    main()
