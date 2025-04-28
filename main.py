#!/usr/bin/env python3

import argparse
import networkx as nx
import sys
import matplotlib.pyplot as plt
import re
import subprocess
from collections import defaultdict
import traceback # Added for detailed error reporting

# Attempt to import cxxfilt, provide instructions if missing
try:
    import cxxfilt
except ImportError:
    print("Error: The 'cxxfilt' library is required for demangling.")
    print("Please install it using: pip install cxxfilt")
    sys.exit(1)

def demangle_name(mangled_name):
    """Demangles C++/Rust names using cxxfilt."""
    if not mangled_name:
        return ""
    name_to_demangle = mangled_name.lstrip('@')
    try:
        # Attempt demangling
        demangled = cxxfilt.demangle(name_to_demangle)
        # Remove Rust hash suffixes (heuristic)
        demangled = re.sub(r'::h[0-9a-f]{16}E$', '', demangled)
        return demangled
    except Exception:
        # Fallback to original name if demangling fails
        return name_to_demangle

def parse_llvm_ir(llvm_code):
    """Parses LLVM IR to find function definitions and call sites."""
    functions = {} # Stores the 'define' line for each function found
    calls = [] # Stores (caller, callee) tuples
    current_function = None
    # Regex to find function definition lines
    func_def_regex = re.compile(r"^\s*define\s+.*\s+(@[^(\s]+)\s*\(")
    # Regex to find call/invoke instructions
    call_regex = re.compile(r"^\s*(?:tail\s+|musttail\s+|notail\s+)?(?:call|invoke)\s+.*\s+(@[^(\s]+)\s*\(")

    lines = llvm_code.splitlines()
    for line in lines:
        line = line.strip()
        func_match = func_def_regex.match(line)
        if func_match:
            # Found a function definition, start tracking its scope
            current_function = func_match.group(1)
            functions[current_function] = line # Store the definition line
            continue # Move to the next line

        if current_function:
            # Inside a function definition, look for calls
            call_match = call_regex.search(line)
            if call_match:
                callee_name = call_match.group(1)
                # Add the call relationship if a valid function context exists
                calls.append((current_function, callee_name))

            # Check if the current function definition ends
            # Use regex to allow optional comments after '}'
            if re.match(r"^\s*\}\s*(;.*)?$", line):
                current_function = None # Exited the current function scope
                
    return functions, calls

def build_call_graph_from_ll(llvm_code):
    """Builds and filters the call graph from parsed LLVM IR."""
    functions, calls = parse_llvm_ir(llvm_code)

    # --- Build the initial full graph ---
    full_graph = nx.DiGraph()
    full_labels = {} # Mangled name -> Demangled name
    all_mangled_names = set(functions.keys())
    all_mangled_names.update(caller for caller, _ in calls)
    all_mangled_names.update(callee for _, callee in calls)

    # Add all identified functions/calls as nodes initially
    for mangled_name in all_mangled_names:
        if not full_graph.has_node(mangled_name):
            full_graph.add_node(mangled_name)
            full_labels[mangled_name] = demangle_name(mangled_name)

    # Add edges based on the parsed calls
    for caller, callee in calls:
        # Ensure both caller and callee nodes exist (they should, based on all_mangled_names)
        if full_graph.has_node(caller) and full_graph.has_node(callee):
            # Avoid self-loops in the basic visualization
            if caller != callee:
                full_graph.add_edge(caller, callee)

    # --- Filter the graph ---
    filtered_graph = nx.DiGraph()
    filtered_display_labels = {} # Mangled name -> Simplified demangled name for display
    print("\nFiltering nodes (keeping non-stdlib, non-closure, non-generic, demangled names):")
    kept_nodes = 0
    filtered_out_nodes = 0

    # Define filtering criteria based on demangled names
    for node, raw_label in full_labels.items():
        keep_node = True
        # --- Filter Rules ---
        # if raw_label == 'main': # Example: Filter out 'main' if desired (currently not filtering main)
        #     keep_node = False
        if raw_label.startswith('_ZN'): # Often internal C++/Rust symbols
             keep_node = False
        elif raw_label.startswith('std::'): # Standard library
             keep_node = False
        elif raw_label.startswith('core::'): # Core library
             keep_node = False
        elif raw_label.startswith('alloc::'): # Alloc library
             keep_node = False
        elif '{{closure}}' in raw_label: # Closures
             keep_node = False
        elif '<' in raw_label or '>' in raw_label: # Generics/Templates
             # Keep intrinsics like llvm.memcpy.<...>
             if not raw_label.startswith("llvm."):
                 keep_node = False
        # --- End Filter Rules ---

        if keep_node:
            # Add the node to the filtered graph
            filtered_graph.add_node(node)
            # Simplify label for display (remove verbose trait paths)
            simplified_label = raw_label
            simplified_label = re.sub(r'<.* as .*>::', '', simplified_label)
            filtered_display_labels[node] = simplified_label
            kept_nodes += 1
        else:
            filtered_out_nodes += 1

    print(f"Kept {kept_nodes} nodes, filtered out {filtered_out_nodes} nodes.")

    # --- Add edges to the filtered graph ---
    print("Adding edges to filtered graph (connecting kept nodes)...")
    added_edges = 0
    kept_node_set = set(filtered_graph.nodes())
    # Iterate through the original graph's edges
    for u, v in full_graph.edges():
        # Add edge only if both source and target nodes are in the kept set
        if u in kept_node_set and v in kept_node_set:
            if not filtered_graph.has_edge(u, v):
                filtered_graph.add_edge(u, v)
                added_edges += 1
    print(f"Added {added_edges} direct edges between kept nodes.")

    # --- Remove isolated nodes from the filtered graph ---
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
    """Detects simple cycles in the graph."""
    cycles_list = []
    cyclic_nodes = set()
    cyclic_edges = set()
    if not graph.nodes:
        return cycles_list, cyclic_nodes, cyclic_edges

    try:
        # Use NetworkX's simple_cycles generator
        # Canonicalize cycles to avoid duplicates due to starting node/rotation
        def canonical_cycle(cycle):
            """Return the minimal rotation of the cycle for canonicalization."""
            if not cycle: return tuple()
            n = len(cycle)
            min_tuple = tuple(cycle)
            for i in range(1, n):
                rotated = tuple(cycle[i:] + cycle[:i])
                if rotated < min_tuple:
                    min_tuple = rotated
            return min_tuple

        seen_cycles = set()
        for cycle in nx.simple_cycles(graph):
            can = canonical_cycle(cycle)
            if can not in seen_cycles:
                seen_cycles.add(can)
                cycles_list.append(list(can)) # Store original cycle order found
                cyclic_nodes.update(can)
                # Identify edges belonging to this cycle
                n = len(can)
                for i in range(n):
                    u = can[i]
                    # Ensure correct edge direction based on original cycle list
                    v_idx = (cycle.index(u) + 1) % len(cycle)
                    v = cycle[v_idx]
                    if graph.has_edge(u, v):
                        cyclic_edges.add((u, v))

    except Exception as e:
        print(f"Error during cycle detection: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return [], set(), set()

    return cycles_list, cyclic_nodes, cyclic_edges


def calculate_edge_cycle_counts(graph, cycles_list):
    """Counts how many simple cycles each edge participates in."""
    edge_counts = defaultdict(int)
    for cycle in cycles_list:
        n = len(cycle)
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            # Check if the edge exists in the graph in this direction
            if graph.has_edge(u, v):
                edge_counts[(u, v)] += 1
    return dict(edge_counts)

def calculate_node_cycle_counts(graph, cycles_list):
    """Counts how many simple cycles each node participates in."""
    node_counts = defaultdict(int)
    for cycle in cycles_list:
        # Use set to count each node only once per cycle
        for node in set(cycle):
            node_counts[node] += 1
    return dict(node_counts)

def estimate_wcet_cycles(function_name, llvm_code):
    """
    Estimate WCET (Worst-Case Execution Time) in cycles for a function
    by analyzing its LLVM IR instructions using a heuristic approach.
    Only the instructions within the function itself are counted; calls
    to other functions are not recursively included.
    Uses a more robust line-by-line method to extract the function body.
    """
    # --- Define approximate cycle costs (heuristic) ---
    costs = {
        # Memory (assume cache miss)
        'load': 100, 'store': 100, 'atomicrmw': 150, 'cmpxchg': 200,
        'fence': 10, 'alloca': 5, 'getelementptr': 2,
        # Integer Arithmetic
        'add': 1, 'sub': 1, 'mul': 5, 'sdiv': 40, 'udiv': 40, 'srem': 40, 'urem': 40,
        # Floating Point Arithmetic
        'fadd': 4, 'fsub': 4, 'fmul': 7, 'fdiv': 40, 'frem': 40,
        # Bitwise & Shift
        'shl': 1, 'lshr': 1, 'ashr': 1, 'and': 1, 'or': 1, 'xor': 1, 'not': 1,
        # Comparison
        'icmp': 1, 'fcmp': 2,
        # Control Flow
        'br': 3, 'switch': 10, 'indirectbr': 15,
        # Calls (Overhead of call itself, not target)
        'call': 40, 'invoke': 50,
        # Return
        'ret': 3,
        # Data Flow
        'phi': 1, 'select': 2,
        # Vector (SIMD)
        'extractelement': 2, 'insertelement': 2, 'shufflevector': 4,
        # Conversion
        'zext': 1, 'sext': 1, 'trunc': 1, 'fptrunc': 1, 'fpext': 1,
        'fptoui': 2, 'fptosi': 2, 'uitofp': 2, 'sitofp': 2,
        'ptrtoint': 1, 'inttoptr': 1, 'bitcast': 1,
        # Memory Intrinsics (assume large, cache miss)
        'memcpy': 200, 'memmove': 200, 'memset': 150,
        # Other
        'unreachable': 0, 'landingpad': 5, # Added cost for landingpad
        'resume': 5, # Added cost for resume
        'unwind': 0, # Exception unwind (not modeled)
    }

    try:
        # --- Revised Function Body Extraction (Line-by-Line) ---
        lines = llvm_code.splitlines()
        function_body_lines = []
        in_function = False
        found_function = False
        # Prepare target function signature check (handle '@')
        target_func_name = function_name.lstrip('@')
        # Regex to precisely match the function definition line start
        define_pattern = re.compile(r"^\s*define\s+.*?\s+@" + re.escape(target_func_name) + r"\s*\(")

        for line in lines:
            stripped_line = line.strip()

            if not in_function:
                # Look for the start of the target function definition
                if define_pattern.match(stripped_line):
                    in_function = True
                    found_function = True
                    # Capture content after '{' if it's on the same line
                    if '{' in stripped_line:
                        body_part = stripped_line.split('{', 1)[1]
                        # Check if the closing '}' is also on this line
                        if re.match(r"^\s*\}\s*(;.*)?$", body_part):
                             # Handle single-line function bodies (like declarations?)
                             # Extract content between { and }
                             single_line_body = body_part.rsplit('}', 1)[0]
                             if single_line_body.strip():
                                 function_body_lines.append(single_line_body)
                             in_function = False # Immediately closed
                        elif body_part.strip():
                             function_body_lines.append(body_part)
                    # Continue to next line to capture rest of body
                    continue
            else: # We are inside the target function
                # Check for the end of the function body '}'
                # Allows optional trailing comments like "; No predecessors"
                if re.match(r"^\s*\}\s*(;.*)?$", stripped_line):
                    in_function = False
                    break # Found end, stop capturing for this function
                else:
                    # Add the line to the current function's body
                    function_body_lines.append(line) # Preserve indentation

        if not found_function:
            # This can happen for external functions (declare) or if name is wrong
            # Check if it's a declaration
            declare_pattern = re.compile(r"^\s*declare\s+.*?\s+@" + re.escape(target_func_name) + r"\s*\(")
            is_declaration = any(declare_pattern.match(l.strip()) for l in lines)
            if is_declaration:
                 # print(f"Info: '{function_name}' is an external declaration.", file=sys.stderr)
                 return 1 # Assign minimal cost for declarations
            else:
                 print(f"Warning: Could not find definition for {function_name}", file=sys.stderr)
                 return None # Function definition truly not found

        if in_function:
             # This indicates a parsing error or malformed LLVM
             print(f"Warning: Reached end of file while still inside function {function_name} (missing '}}'?)", file=sys.stderr)
             # Proceeding with partial body might give wrong results, return None
             return None

        # Join the collected lines to form the function body string
        function_body = "\n".join(function_body_lines)

        # Handle cases where body might be empty after extraction (e.g., formatting issues)
        if not function_body.strip() and found_function:
             # print(f"Warning: Extracted empty body for defined function {function_name}. Assigning minimal cost.", file=sys.stderr)
             return 5 # Assign slightly more than declaration

        # --- End Revised Function Body Extraction ---


        # --- Instruction Counting (on the extracted body) ---
        estimated_cycles = 0
        instruction_count = 0
        # Use the extracted body lines for analysis
        body_lines_for_analysis = function_body.splitlines()

        for line in body_lines_for_analysis:
            stripped_line = line.strip()
            # Skip empty lines, comments, labels
            if not stripped_line or stripped_line.startswith(';') or stripped_line.endswith(':'):
                continue

            # Extract opcode (first word after optional assignment like '%res =')
            match_opcode = re.match(r'\s*(?:%[\w.]+\s*=\s*)?(\w+)', stripped_line)
            if match_opcode:
                opcode = match_opcode.group(1)
                # Accumulate cost if opcode is known
                if opcode in costs:
                    estimated_cycles += costs[opcode]
                    instruction_count += 1
                # else: # Optional: Warn about unknown opcodes
                #     print(f"Warning: Unknown opcode '{opcode}' in {function_name}", file=sys.stderr)

        # --- Complexity Factors ---
        # Basic Blocks: Count labels (like 'bb1:') within the function body
        # Add 1 for the implicit entry block
        basic_blocks = len(re.findall(r'^\s*[\w.]+:', function_body, re.MULTILINE)) + 1
        estimated_cycles += basic_blocks * 1 # Add 1 cycle per block for potential jumps

        # --- Minimum Cost ---
        min_cost = 5 + instruction_count # Base cost + 1 per instruction found
        estimated_cycles = max(estimated_cycles, min_cost)

        # --- Loop Heuristic (Basic Back-Edge Counting) ---
        labels = {}
        back_edges = 0
        # Find all labels and their line numbers within the extracted body
        for i, line in enumerate(body_lines_for_analysis):
             # Match labels like 'entry:', 'bb1:', 'loop.header:' etc.
             label_match = re.match(r'^\s*([\w.]+):', line.strip())
             if label_match:
                 labels[label_match.group(1)] = i # Store label name and line number

        # Find branches to labels defined earlier textually
        for i, line in enumerate(body_lines_for_analysis):
            # Match 'br label %target_label'
            branch_match = re.search(r'\bbr\s+label\s+%([\w.]+)', line)
            if branch_match:
                target_label = branch_match.group(1)
                # Check if the target label exists and is defined on an earlier line
                if target_label in labels and labels[target_label] < i:
                    back_edges += 1 # Found a potential back-edge

        # Apply penalty if potential loops are detected
        if back_edges > 0:
             # Penalty proportional to back-edges and estimated base cost
             loop_penalty = back_edges * (estimated_cycles * 0.5) # Heuristic factor
             estimated_cycles += int(loop_penalty)

        return estimated_cycles # Return the final estimated cycle count

    except Exception as e:
        # Print detailed error information if estimation fails
        print(f"Error estimating cycles for {function_name}: {e}", file=sys.stderr)
        demangled_name = demangle_name(function_name)
        print(f"  - {demangled_name}: Unable to estimate cycles", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Print stack trace
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
    # node_cycle_counts = calculate_node_cycle_counts(graph, cycles_list) # Currently unused in drawing

    # --- Calculate WCET estimates for each node ---
    wcet_estimates = {}
    print("Estimating WCET cycles for each function (independent, not including callees)...")
    # Use the provided display labels (already simplified) for printing progress
    for node, display_label in labels.items():
        # Pass the original mangled node name to the estimation function
        wcet_cycles = estimate_wcet_cycles(node, llvm_code)
        if wcet_cycles is not None: # Check for None in case of estimation error
            wcet_estimates[node] = wcet_cycles
            print(f"  - {display_label}: ~{wcet_cycles} cycles")
        else:
             print(f"  - {display_label}: WCET estimation failed")


    # --- Graph Drawing Setup ---
    num_nodes = len(graph.nodes)
    # Adjust figure size based on number of nodes for better layout
    fig_width = max(18, num_nodes * 1.3)
    fig_height = max(15, num_nodes * 1.1)
    plt.figure(figsize=(fig_width, fig_height))

    # Spring layout parameters (adjust k for spacing)
    k_value = 5.0 / (num_nodes**0.5) if num_nodes > 1 else 5.0
    k_value = max(k_value, 0.7) # Ensure minimum spacing
    iterations = 200 # More iterations for potentially better layout

    # Visual parameters
    base_node_size = 2200
    cyclic_node_size = 2600 # Make cyclic nodes slightly larger
    font_size = 10 # Smaller font for potentially many nodes
    arrow_size = 35
    connection_radius = 0.3 # Curve for edges

    # --- Calculate Node Positions ---
    try:
        print(f"Calculating spring layout for filtered graph (k={k_value:.2f}, iterations={iterations})...")
        # Use a fixed seed for reproducible layouts
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

    # --- Prepare Node/Edge Styles ---
    node_list = list(graph.nodes)
    edge_list = list(graph.edges)

    node_colors = ["#d0d0ff"] * len(node_list) # Light blue/purple nodes
    edge_colors = ["#999999"] * len(edge_list) # Grey edges
    edge_widths = [1.5] * len(edge_list)
    node_sizes = [base_node_size] * len(node_list)
    node_edge_colors = ["#444444"] * len(node_list) # Dark grey node borders

    # Highlight cyclic nodes
    if cyclic_nodes:
        for i, node in enumerate(node_list):
            if node in cyclic_nodes:
                node_colors[i] = "#ffcccc" # Light red cyclic nodes
                node_sizes[i] = cyclic_node_size
                node_edge_colors[i] = "#cc0000" # Dark red border

    # Highlight cyclic edges
    if cyclic_edges:
        for i, edge in enumerate(edge_list):
            if edge in cyclic_edges:
                edge_colors[i] = "#ff0000" # Red cyclic edges
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
            edgecolors=node_edge_colors,
            linewidths=2.0 # Slightly thinner border
        )

        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=edge_list,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowstyle='-|>', # Standard arrow head
            arrowsize=arrow_size,
            connectionstyle=f'arc3,rad={connection_radius}', # Curved edges
            min_source_margin=25, # Margin around node for arrow start/end
            min_target_margin=25,
            alpha=0.8 # Slight transparency
        )

        # Draw node labels (using the pre-simplified display labels)
        nx.draw_networkx_labels(
            graph, pos, labels=labels, font_size=font_size, font_family="sans-serif", font_weight="bold", font_color="#111111"
        )

        # --- Draw WCET Estimates Below Nodes ---
        for node in node_list:
            if node not in pos: continue # Skip if layout failed for node
            x, y = pos[node]
            # Adjust vertical offset based on figure size or node size if needed
            vertical_offset = -0.06 * (fig_height / 15.0) # Scale offset slightly with figure height

            # Prepare label text
            if node in wcet_estimates:
                wcet = wcet_estimates[node]
                label_text = f"WCET: ~{wcet}" # Shorter label
            else:
                label_text = "WCET: N/A" # Indicate if estimation failed

            # Draw the WCET label with a background box
            plt.text(
                x, y + vertical_offset, # Position below the node center
                label_text,
                fontsize=font_size - 1, # Slightly smaller than main label
                ha='center', # Horizontal alignment
                va='top',    # Vertical alignment
                color='#0000cc', # Blue text
                bbox=dict(
                    facecolor='#ffffff', # White background
                    edgecolor='#aaaaaa', # Light grey border
                    boxstyle='round,pad=0.2', # Rounded box with padding
                    linewidth=1.0,
                    alpha=0.85 # Slightly transparent background
                ),
                zorder=10 # Ensure text is drawn on top
            )

        # --- Draw Edge Cycle Counts (Optional) ---
        # This can clutter the graph, consider enabling only if needed
        # for i, (u, v) in enumerate(edge_list):
        #     if (u, v) in edge_cycle_counts and edge_cycle_counts[(u, v)] > 0:
        #         # Calculate midpoint, handle curved edges if necessary
        #         # ... logic to place text near edge midpoint ...
        #         count = edge_cycle_counts[(u,v)]
        #         # plt.text(mid_x, mid_y, str(count), ...)


        # --- Final Touches & Save ---
        file_base_name = svg_path.rsplit('.', 1)[0] # Get name without extension
        plt.title(f"Filtered LLVM IR Call Graph (Cycles Highlighted)\nSource: {file_base_name}.ll", fontsize=16)
        plt.axis("off") # Hide axes
        plt.tight_layout(pad=1.0) # Adjust padding
        print(f"Saving filtered graph to {svg_path}...")
        plt.savefig(svg_path, format="svg", bbox_inches='tight', dpi=150) # Save as SVG
        print(f"\nFiltered call graph exported as SVG: {svg_path}")

    except Exception as draw_error:
        print(f"Error during graph drawing or saving: {draw_error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        # Ensure the plot is closed to free memory
        plt.close()

def main():
    # --- Argument Parsing ---
    parser_arg = argparse.ArgumentParser(
        description="Analyze LLVM IR code (.ll), filter graph to user functions, detect cycles, demangle names, estimate WCET, and export as SVG."
    )
    parser_arg.add_argument("llvm_file", help="Path to the LLVM IR source file (.ll).")
    parser_arg.add_argument(
        "--svg", dest="svg_file", default=None,
        help="Path to output SVG file (default: <llvm_file_base>.filtered.svg)"
    )
    args = parser_arg.parse_args()

    # --- Read LLVM File ---
    try:
        with open(args.llvm_file, "r", encoding="utf-8") as f:
            llvm_code = f.read()
    except FileNotFoundError:
        print(f"Error: File not found '{args.llvm_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{args.llvm_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Build and Filter Graph ---
    print(f"Parsing LLVM IR file: {args.llvm_file}...")
    try:
        # Pass the full LLVM code to the graph builder
        call_graph, display_labels = build_call_graph_from_ll(llvm_code)
    except Exception as e:
        print(f"Error building or filtering call graph: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # Check if graph is empty after filtering
    if not call_graph.nodes:
        print("\nNo functions matching the filter criteria were found or remained after filtering.")
        sys.exit(0)
    else:
        print(f"\nFiltered graph contains {len(call_graph.nodes)} functions and {len(call_graph.edges)} calls.")

    # --- Detect Cycles ---
    print("Detecting cycles in filtered graph...")
    cycles_list, cyclic_nodes, cyclic_edges = detect_cycles(call_graph)
    num_cycles = len(cycles_list)

    if cyclic_nodes:
        # Sort nodes for consistent output
        sorted_mangled_cyclic_nodes = sorted(list(cyclic_nodes))
        print(f"\nDetected {num_cycles} simple cycle(s) involving {len(sorted_mangled_cyclic_nodes)} function(s):")
        for mangled_node in sorted_mangled_cyclic_nodes:
            # Use the display label if available, otherwise fallback to mangled name
            display_name = display_labels.get(mangled_node, mangled_node)
            print(f"  - {display_name} ({mangled_node})") # Show both for clarity
    else:
        print("\nNo cycles detected in the filtered call graph.")

    # --- Determine Output SVG Path ---
    svg_path = args.svg_file
    if not svg_path:
        # Default output path based on input filename
        base_name = args.llvm_file
        if base_name.lower().endswith(".ll"):
            base_name = base_name[:-3] # Remove .ll extension
        svg_path = base_name + ".filtered.svg"

    # --- Generate Visualization ---
    print(f"\nGenerating filtered graph visualization...")
    # Pass the full llvm_code again for WCET estimation within the drawing function
    draw_and_export_graph(
        call_graph,
        display_labels,
        svg_path,
        cycles_list,
        llvm_code, # Pass the original LLVM code
        cyclic_nodes=cyclic_nodes,
        cyclic_edges=cyclic_edges
    )

if __name__ == "__main__":
    main()
