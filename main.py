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
    if not mangled_name:
        return ""
    name_to_demangle = mangled_name.lstrip('@')
    try:
        demangled = cxxfilt.demangle(name_to_demangle)
        demangled = re.sub(r'::h[0-9a-f]{16}E$', '', demangled)
        return demangled
    except Exception:
        return name_to_demangle

def parse_llvm_ir(llvm_code):
    functions = {}
    calls = []
    current_function = None
    func_def_regex = re.compile(r"^\s*define\s+.*\s+(@[^(\s]+)\s*\(")
    call_regex = re.compile(r"^\s*(?:tail\s+|musttail\s+|notail\s+)?(?:call|invoke)\s+.*\s+(@[^(\s]+)\s*\(")
    lines = llvm_code.splitlines()
    for line in lines:
        line = line.strip()
        func_match = func_def_regex.match(line)
        if func_match:
            current_function = func_match.group(1)
            functions[current_function] = line
            continue
        if current_function:
            call_match = call_regex.search(line)
            if call_match:
                callee_name = call_match.group(1)
                calls.append((current_function, callee_name))
            if line == "}":
                current_function = None
    return functions, calls

def build_call_graph_from_ll(llvm_code):
    functions, calls = parse_llvm_ir(llvm_code)
    full_graph = nx.DiGraph()
    full_labels = {}
    all_mangled_names = set(functions.keys())
    all_mangled_names.update(caller for caller, _ in calls)
    all_mangled_names.update(callee for _, callee in calls)
    for mangled_name in all_mangled_names:
        if not full_graph.has_node(mangled_name):
            full_graph.add_node(mangled_name)
            full_labels[mangled_name] = demangle_name(mangled_name)
    for caller, callee in calls:
        if full_graph.has_node(caller) and full_graph.has_node(callee):
            if caller != callee:
                full_graph.add_edge(caller, callee)
    filtered_graph = nx.DiGraph()
    filtered_display_labels = {}
    print("\nFiltering nodes (keeping non-stdlib, non-closure, non-generic, demangled names):")
    kept_nodes = 0
    filtered_out_nodes = 0
    for node, raw_label in full_labels.items():
        keep_node = True
        if raw_label == 'main':
            keep_node = False
        elif raw_label.startswith('_ZN'):
            keep_node = False
        elif raw_label.startswith('std::'):
            keep_node = False
        elif raw_label.startswith('core::'):
            keep_node = False
        elif raw_label.startswith('alloc::'):
            keep_node = False
        elif '{{closure}}' in raw_label:
            keep_node = False
        elif '<' in raw_label or '>' in raw_label:
            keep_node = False
        if keep_node:
            filtered_graph.add_node(node)
            simplified_label = raw_label
            simplified_label = re.sub(r'<.* as .*>::', '', simplified_label)
            filtered_display_labels[node] = simplified_label
            kept_nodes += 1
        else:
            filtered_out_nodes += 1
    print(f"Kept {kept_nodes} nodes, filtered out {filtered_out_nodes} nodes.")
    print("Adding edges to filtered graph (connecting kept nodes)...")
    added_edges = 0
    kept_node_set = set(filtered_graph.nodes())
    for u in kept_node_set:
        for v in full_graph.successors(u):
            if v in kept_node_set:
                if not filtered_graph.has_edge(u, v):
                    filtered_graph.add_edge(u, v)
                    added_edges += 1
    print(f"Added {added_edges} direct edges between kept nodes.")
    isolated_nodes = list(nx.isolates(filtered_graph))
    if isolated_nodes:
        print(f"Removing {len(isolated_nodes)} isolated nodes after filtering and edge addition.")
        filtered_graph.remove_nodes_from(isolated_nodes)
        for node in isolated_nodes:
            if node in filtered_display_labels:
                del filtered_display_labels[node]
    return filtered_graph, filtered_display_labels

def detect_cycles(graph):
    cycles_list = []
    cyclic_nodes = set()
    cyclic_edges = set()
    if not graph.nodes:
        return cycles_list, cyclic_nodes, cyclic_edges
    try:
        cycles_list = list(nx.simple_cycles(graph))
        for cycle in cycles_list:
            cyclic_nodes.update(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if graph.has_edge(u, v):
                    cyclic_edges.add((u, v))
    except Exception as e:
        print(f"Error during cycle detection: {e}", file=sys.stderr)
        return [], set(), set()
    return cycles_list, cyclic_nodes, cyclic_edges

def calculate_edge_cycle_counts(graph, cycles_list):
    edge_counts = defaultdict(int)
    for cycle in cycles_list:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if graph.has_edge(u, v):
                edge_counts[(u, v)] += 1
    return dict(edge_counts)

def calculate_node_cycle_counts(graph, cycles_list):
    """
    Returns a dict mapping node to the number of cycles it participates in.
    """
    node_counts = defaultdict(int)
    for cycle in set(cycles_list):
        for node in set(cycle):
            node_counts[node] += 1
    return dict(node_counts)

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
    print("Estimating WCET cycles for each function...")
    for node in graph.nodes:
        wcet_cycles = estimate_wcet_cycles(node, llvm_code)
        if wcet_cycles:
            wcet_estimates[node] = wcet_cycles
            print(f"  - {labels.get(node, node)}: ~{wcet_cycles} cycles")

    # --- Adjustments for better spacing and visibility ---
    num_nodes = len(graph.nodes)
    fig_width = max(18, num_nodes * 1.2)
    fig_height = max(15, num_nodes * 1.0)
    plt.figure(figsize=(fig_width, fig_height))

    k_value = 4.5 / (num_nodes**0.5) if num_nodes > 1 else 4.5
    k_value = max(k_value, 0.6)
    iterations = 200

    base_node_size = 2000
    cyclic_node_size = 2400
    font_size = 11
    arrow_size = 40
    connection_radius = 0.35

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

    node_colors = ["#ccccff"] * len(graph.nodes)
    edge_colors = ["#aaaaaa"] * len(graph.edges)
    edge_widths = [1.5] * len(graph.edges)
    node_sizes = [base_node_size] * len(graph.nodes)
    node_edge_colors = ["#333333"] * len(graph.nodes)

    node_list = list(graph.nodes)
    edge_list = list(graph.edges)

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
                edge_widths[i] = 3.5

    try:
        print("Drawing filtered graph elements...")
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=node_list,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors=node_edge_colors,
            linewidths=2.5
        )

        nx.draw_networkx_edges(
            graph, pos,
            edgelist=edge_list,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=arrow_size,
            connectionstyle=f'arc3,rad={connection_radius}',
            min_source_margin=25,
            min_target_margin=25,
            alpha=0.95
        )

        # Draw node labels (function names)
        nx.draw_networkx_labels(
            graph, pos, labels=labels, font_size=font_size, font_family="monospace", font_color="#222222"
        )

        # Draw WCET cycle estimates under each node
        for node in node_list:
            x, y = pos[node]
            node_size = node_sizes[node_list.index(node)]
            vertical_offset = -0.08
            if num_nodes > 10:
                vertical_offset = -0.10
            # Compose the label with WCET estimate
            if node in wcet_estimates:
                wcet = wcet_estimates[node]
                label_text = f"Est. WCET: {wcet} cycles"
            else:
                label_text = "Est. WCET: unknown"
            # Draw the label with a nice background and padding
            plt.text(
                x, y + vertical_offset,
                label_text,
                fontsize=font_size + 2,
                ha='center',
                va='top',
                color='blue',
                bbox=dict(
                    facecolor='white',
                    edgecolor='blue',
                    boxstyle='round,pad=0.25',
                    linewidth=1.3,
                    alpha=0.95
                ),
                zorder=10
            )

        # Draw custom arrows in the middle of edges with cycle counts (optional, can be removed)
        for i, (u, v) in enumerate(edge_list):
            if (u, v) in edge_cycle_counts and edge_cycle_counts[(u, v)] > 0:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                mid_x = x1 + 0.5 * (x2 - x1)
                mid_y = y1 + 0.5 * (y2 - y1)
                dx = x2 - x1
                dy = y2 - y1
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    dx, dy = dx/length, dy/length
                if connection_radius != 0:
                    perpx, perpy = -dy, dx
                    mid_x += connection_radius * perpx * 6
                    mid_y += connection_radius * perpy * 6
                arrow_color = 'blue' if (cyclic_edges and (u, v) in cyclic_edges) else 'black'
                plt.annotate(
                    '', xy=(mid_x + dx*0.08, mid_y + dy*0.08), xytext=(mid_x, mid_y),
                    arrowprops=dict(
                        arrowstyle='-|>,head_width=0.7,head_length=1.2',
                        color=arrow_color,
                        lw=2.5,
                        shrinkA=0, shrinkB=0,
                        alpha=1.0
                    ),
                    annotation_clip=False
                )

        # Legend removed as per instructions

        file_base_name = svg_path.replace('.svg', '.ll')
        plt.title(f"Filtered LLVM IR Call Graph (User Functions) - Cycles Highlighted\nFile: {file_base_name}", fontsize=15)
        plt.axis("off")
        plt.tight_layout()
        print(f"Saving filtered graph to {svg_path}...")
        plt.savefig(svg_path, format="svg", bbox_inches='tight', dpi=180)
        print(f"\nFiltered call graph exported as SVG: {svg_path}")

    except Exception as draw_error:
        print(f"Error during graph drawing or saving: {draw_error}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        plt.close()

def main():
    parser_arg = argparse.ArgumentParser(
        description="Analyze LLVM IR code (.ll), filter graph to user functions, detect cycles, demangle names, and export as SVG with WCET cycle estimates under each node."
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
        call_graph, display_labels = build_call_graph_from_ll(llvm_code)
    except Exception as e:
        print(f"Error building or filtering call graph: {e}", file=sys.stderr)
        sys.exit(1)

    if not call_graph.nodes:
        print("\nNo functions matching the filter criteria were found in the provided LLVM IR code.")
        sys.exit(0)
    else:
        print(f"\nFiltered graph contains {len(call_graph.nodes)} functions and {len(call_graph.edges)} calls.")

    print("Detecting cycles in filtered graph...")
    cycles_list, cyclic_nodes, cyclic_edges = detect_cycles(call_graph)
    num_cycles = len(cycles_list)

    if cyclic_nodes:
        sorted_mangled_cyclic_nodes = sorted(list(cyclic_nodes))
        print(f"\nDetected {num_cycles} simple cycle(s) involving {len(sorted_mangled_cyclic_nodes)} function(s) within the filtered graph:")
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
        svg_path = base_name + ".filtered.svg"

    print(f"\nGenerating filtered graph visualization...")
    draw_and_export_graph(call_graph, display_labels, svg_path, cycles_list, llvm_code, cyclic_nodes=cyclic_nodes, cyclic_edges=cyclic_edges)

if __name__ == "__main__":
    main()
