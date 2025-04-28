#!/usr/bin/env python3

import argparse
import networkx as nx
import sys
import matplotlib.pyplot as plt
import re
import subprocess
import cxxfilt
import traceback

from collections import defaultdict

# Estimated costs based on typical x86-64 latencies (highly approximate)
# Specific to instructions found in linear.s

assembly_costs = {
    # Stack Operations
    "pushq": 1,  # Typically fast, might involve micro-ops for stack engine
    "popq": 1,   # Typically fast

    # Data Movement (Register/Immediate focused)
    # Note: MOV involving memory (e.g., movq %rdi, (%rsp)) cost depends heavily on cache.
    # Assuming ~1 cycle for reg-reg/imm-reg, higher for memory. We use a base cost here.
    "movl": 1,
    "movq": 1,
    "movabsq": 1, # Move 64-bit immediate, usually fast
    "movzbl": 1,  # Zero-extend byte to long, usually fast
    "movslq": 1,  # Sign-extend long to quad, usually fast
    "leaq": 1,   # Address calculation, usually 1 cycle latency

    # Control Flow
    "callq": 5,  # Mechanics of call/jump + potential pipeline effects. Indirect calls can be costlier. Cost of target function is separate.
    "retq": 3,   # Mechanics of return, potential branch prediction interaction.
    "jmp": 1,    # Unconditional jump (if predicted correctly). Mispredict penalty is high (~15-20+).
    "jo": 1,     # Conditional jump (if predicted correctly). Mispredict penalty is high (~15-20+).

    # Arithmetic / Logical
    "addq": 1,
    "subq": 1,   # Integer subtraction, typically 1 cycle latency
    "xorl": 1,   # Bitwise XOR (often used to zero registers), typically 1 cycle latency
    "incl": 1,   # Increment, typically 1 cycle latency

    # Status Flags
    "seto": 1,   # Set byte based on flag, typically fast
    
    # --- Placeholder for other common instructions (NOT explicitly in linear.s, but often relevant) ---
    # "addq": 1, 
    # "imulq": 3-5, # Integer Multiply
    # "idivq": 10-40, # Integer Divide (highly variable)
    # "cmpq": 1, # Comparison
    # "testq": 1, # Bitwise test
    # --- Memory explicit variants ---
    # "mov_mem_load": 100, # Example cost assuming L3 miss (as per your earlier WCET idea)
    # "mov_mem_store": 100, # Example cost assuming L3 miss
}

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
            if re.match(r"^\s*\}\s*(;.*)?$", line):
                current_function = None

    return functions, calls


def build_call_graph_from_ll(llvm_code, no_filter=False):
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
    if no_filter:
        print("\n--no-filter specified: No filtering of nodes will be performed.")
        return full_graph, full_labels
    filtered_graph = nx.DiGraph()
    filtered_display_labels = {}
    print("\nFiltering nodes (keeping non-stdlib, non-closure, non-generic, demangled names):")
    kept_nodes = 0
    filtered_out_nodes = 0
    for node, raw_label in full_labels.items():
        keep_node = True
        if raw_label.startswith("_ZN"):
            keep_node = False
        elif raw_label.startswith("std::"):
            keep_node = False
        elif raw_label.startswith("core::"):
            keep_node = False
        elif raw_label.startswith("alloc::"):
            keep_node = False
        elif "{{closure}}" in raw_label:
            keep_node = False
        elif "<" in raw_label or ">" in raw_label:
            if not raw_label.startswith("llvm."):
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
    for u, v in full_graph.edges():
        if u in kept_node_set and v in kept_node_set:
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

        def canonical_cycle(cycle):
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
                cycles_list.append(list(can))
                cyclic_nodes.update(can)
                n = len(can)
                for i in range(n):
                    u = can[i]
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
    edge_counts = defaultdict(int)
    for cycle in cycles_list:
        n = len(cycle)
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            if graph.has_edge(u, v):
                edge_counts[(u, v)] += 1
    return dict(edge_counts)


def calculate_node_cycle_counts(graph, cycles_list):
    node_counts = defaultdict(int)
    for cycle in cycles_list:
        for node in set(cycle):
            node_counts[node] += 1
    return dict(node_counts)


def estimate_wcet_cycles(function_name, assembly_code):
    """
    Estimates WCET cycles for a function based on its assembly code.

    Args:
        function_name: The mangled name of the function (e.g., '@_ZN...').
        assembly_code: A string containing the entire assembly code.

    Returns:
        An integer representing the estimated cycle count, or None if error.
    """
    # Use the global assembly_costs dictionary
    global assembly_costs

    # Normalize function name (remove leading '@', common in LLVM IR but maybe not assembly labels)
    # Assembly labels might have a leading underscore or other prefixes depending on the ABI/platform.
    base_func_name = function_name.lstrip('@')
    possible_labels = [
        f"{base_func_name}:",         # e.g., main:
        f"_{base_func_name}:",        # e.g., _main:
        f".{base_func_name}:"         # e.g., .main: (less common for functions)
    ]
    # Add common linkage names if needed, e.g., from .globl directive
    possible_globls = [
        f".globl\t{base_func_name}",
        f".globl\t_{base_func_name}"
    ]


    estimated_cycles = 0
    instruction_count = 0
    in_function = False
    found_function = False

    lines = assembly_code.splitlines()

    try:
        for i, line in enumerate(lines):
            stripped_line = line.strip()

            if not in_function:
                # Check if the line matches any possible start label
                if any(stripped_line.startswith(label) for label in possible_labels):
                    # Check if this label was declared global nearby
                    is_global = False
                    for j in range(max(0, i - 5), i): # Check previous lines for .globl
                        if any(glob_pattern in lines[j] for glob_pattern in possible_globls):
                            is_global = True
                            break
                    # Crude check: if it looks like a function label and is maybe global, start parsing
                    if is_global or ':' in stripped_line: # Basic check
                         in_function = True
                         found_function = True
                         print(f"INFO: Starting WCET count for {function_name} at line {i+1}: {stripped_line}")

            if in_function:
                # Remove comments first (handle '#' and potentially ';')
                line_no_comment = stripped_line.split('#', 1)[0].split(';', 1)[0].strip()

                # Stop if we hit another likely function label or end directive
                # Check the line *without* comments
                if line_no_comment.endswith(':') and not any(line_no_comment == label for label in possible_labels) and instruction_count > 0:
                    print(f"INFO: Ending WCET count for {function_name} before line {i+1}: {stripped_line}")
                    in_function = False
                    break
                if line_no_comment.startswith(".cfi_endproc"):
                    print(f"INFO: Ending WCET count for {function_name} at CFI directive line {i+1}")
                    in_function = False
                    break

                # Ignore directives, empty lines, or labels based on the comment-free line
                if line_no_comment.startswith('.') or not line_no_comment or \
                    line_no_comment.endswith(':'):
                    continue


                # Extract the instruction mnemonic from the comment-free line
                parts = line_no_comment.split(None, 1) # Split on first whitespace
                if parts:
                    mnemonic = parts[0].lower() # Use lower case for matching keys
                    cost = assembly_costs.get(mnemonic)

                    if cost is not None:
                        estimated_cycles += cost
                        instruction_count += 1
                    else:
                        print(f"Warning: Unknown assembly instruction '{mnemonic}' in function {function_name}. Assigning default cost 1.", file=sys.stderr)
                        estimated_cycles += 1
                        instruction_count += 1
                if stripped_line.endswith(':') and not any(stripped_line == label for label in possible_labels) and instruction_count > 0:
                    print(f"INFO: Ending WCET count for {function_name} before line {i+1}: {stripped_line}")
                    in_function = False
                    break
                if stripped_line.startswith(".cfi_endproc"):
                    print(f"INFO: Ending WCET count for {function_name} at CFI directive line {i+1}")
                    in_function = False
                    break

                # Ignore comments, directives, empty lines, labels
                if stripped_line.startswith('#') or stripped_line.startswith(';') or \
                   stripped_line.startswith('.') or not stripped_line or \
                   stripped_line.endswith(':'):
                    continue

                # Extract the instruction mnemonic
                parts = stripped_line.split(None, 1) # Split on first whitespace
                if parts:
                    mnemonic = parts[0].lower() # Use lower case for matching keys
                    cost = assembly_costs.get(mnemonic)

                    if cost is not None:
                        estimated_cycles += cost
                        instruction_count += 1
                    else:
                        # Optional: Add a default cost or warning for unknown instructions
                        print(f"Warning: Unknown assembly instruction '{mnemonic}' in function {function_name}. Assigning default cost 1.", file=sys.stderr)
                        estimated_cycles += 1
                        instruction_count += 1


        if not found_function:
            # Function might be external or non-existent in this assembly
            # Check if it's declared (common for library calls)
            if any(f".type\t{base_func_name}, @function" in line for line in lines) or \
               any(f".globl\t{base_func_name}" in line for line in lines) or \
               any(f".symver {base_func_name}," in line for line in lines):
                 print(f"INFO: Function {function_name} seems declared but not defined. Assigning minimal cost 1.")
                 return 1
            else:
                print(f"Warning: Could not find assembly definition label for {function_name}", file=sys.stderr)
                return None

        # Basic minimum cost
        min_cost = 1 + instruction_count
        estimated_cycles = max(estimated_cycles, min_cost)

        return estimated_cycles

    except Exception as e:
        print(f"Error estimating assembly cycles for {function_name}: {e}", file=sys.stderr)
        demangled_name = demangle_name(function_name)
        print(f"  - {demangled_name}: Unable to estimate assembly cycles", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def draw_and_export_graph(
    graph,
    labels,
    svg_path,
    cycles_list,
    assembly_code,
    cyclic_nodes=None,
    cyclic_edges=None,
):
    if not graph.nodes:
        print("Filtered graph is empty, cannot draw.", file=sys.stderr)
        return

    edge_cycle_counts = calculate_edge_cycle_counts(graph, cycles_list)
    wcet_estimates = {}
    print(
        "Estimating WCET cycles for each function from Assembly (independent, not including callees)..."
    )
    for node, display_label in labels.items():
        wcet_cycles = estimate_wcet_cycles(node, assembly_code)
        if wcet_cycles is not None:
            wcet_estimates[node] = wcet_cycles
            print(f"  - {display_label}: ~{wcet_cycles} cycles (from assembly)")
        else:
            print(f"  - {display_label}: Assembly WCET estimation failed")

    num_nodes = len(graph.nodes)
    fig_width = max(18, num_nodes * 1.3)
    fig_height = max(15, num_nodes * 1.1)
    plt.figure(figsize=(fig_width, fig_height))
    k_value = 5.0 / (num_nodes**0.5) if num_nodes > 1 else 5.0
    k_value = max(k_value, 0.7)
    iterations = 200
    base_node_size = 2200
    cyclic_node_size = 2600
    font_size = 10
    arrow_size = 35
    connection_radius = 0.3
    try:
        print(
            f"Calculating spring layout for filtered graph (k={k_value:.2f}, iterations={iterations})..."
        )
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
    node_list = list(graph.nodes)
    edge_list = list(graph.edges)
    node_colors = ["#d0d0ff"] * len(node_list)
    edge_colors = ["#999999"] * len(edge_list)
    edge_widths = [1.5] * len(edge_list)
    node_sizes = [base_node_size] * len(node_list)
    node_edge_colors = ["#444444"] * len(node_list)
    if cyclic_nodes:
        for i, node in enumerate(node_list):
            if node in cyclic_nodes:
                node_colors[i] = "#ffcccc"
                node_sizes[i] = cyclic_node_size
                node_edge_colors[i] = "#cc0000"
    if cyclic_edges:
        for i, edge in enumerate(edge_list):
            if edge in cyclic_edges:
                edge_colors[i] = "#ff0000"
                edge_widths[i] = 3.0
    try:
        print("Drawing filtered graph elements...")
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=node_list,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors=node_edge_colors,
            linewidths=2.0,
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edge_list,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=arrow_size,
            connectionstyle=f"arc3,rad={connection_radius}",
            min_source_margin=25,
            min_target_margin=25,
            alpha=0.8,
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            labels=labels,
            font_size=font_size,
            font_family="sans-serif",
            font_weight="bold",
            font_color="#111111",
        )
        for node in node_list:
            if node not in pos:
                continue
            x, y = pos[node]
            vertical_offset = -0.06 * (fig_height / 15.0)
            if node in wcet_estimates:
                wcet = wcet_estimates[node]
                label_text = f"ASM Cycles: ~{wcet}"
            else:
                label_text = "ASM Cycles: N/A"
            plt.text(
                x,
                y + vertical_offset,
                label_text,
                fontsize=font_size - 1,
                ha="center",
                va="top",
                color="#0000cc",
                bbox=dict(
                    facecolor="#ffffff",
                    edgecolor="#aaaaaa",
                    boxstyle="round,pad=0.2",
                    linewidth=1.0,
                    alpha=0.85,
                ),
                zorder=10,
            )
        file_base_name = svg_path.rsplit(".", 1)[0]
        # Consider updating the title if the primary source is now assembly
        plt.title(f"Filtered LLVM IR Call Graph (Assembly Cycles Estimated)\nSource LLVM: {file_base_name}.ll", fontsize=16)
        plt.axis("off")
        plt.tight_layout(pad=1.0)
        print(f"Saving filtered graph to {svg_path}...")
        plt.savefig(svg_path, format="svg", bbox_inches="tight", dpi=150)
        print(f"\nFiltered call graph exported as SVG: {svg_path}")
    except Exception as draw_error:
        print(f"Error during graph drawing or saving: {draw_error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        plt.close()


def main():
    parser_arg = argparse.ArgumentParser(
        description="Analyze LLVM IR call graph, estimate WCET from Assembly, detect cycles, and export SVG."
    )
    parser_arg.add_argument("llvm_file", help="Path to the LLVM IR source file (.ll).")
    parser_arg.add_argument(
        "--asm-file",
        dest="asm_file",
        required=True,
        help="Path to the corresponding Assembly source file (.s)."
    )
    parser_arg.add_argument(
        "--svg", dest="svg_file", default=None,
        help="Path to output SVG file (default: <llvm_file_base>.filtered.svg)"
    )
    parser_arg.add_argument(
        "--no-filter",
        dest="no_filter",
        action="store_true",
        help="Disable all filtering of functions; include all nodes and edges in the call graph.",
    )
    args = parser_arg.parse_args()

    # Read LLVM IR file
    try:
        with open(args.llvm_file, "r", encoding="utf-8") as f:
            llvm_code = f.read()
    except FileNotFoundError:
        print(f"Error: LLVM file not found '{args.llvm_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading LLVM file '{args.llvm_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Read Assembly file
    try:
        with open(args.asm_file, "r", encoding="utf-8") as f:
            assembly_code = f.read()
    except FileNotFoundError:
        print(f"Error: Assembly file not found '{args.asm_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading Assembly file '{args.asm_file}': {e}", file=sys.stderr)
        sys.exit(1)


    print(f"Parsing LLVM IR file: {args.llvm_file}...")
    try:
        # Build call graph from LLVM (structure comes from IR)
        call_graph, display_labels = build_call_graph_from_ll(
            llvm_code, no_filter=args.no_filter
        )
    except Exception as e:
        print(f"Error building or filtering call graph: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    if not call_graph.nodes:
        if args.no_filter:
            print("\nNo functions found in the LLVM IR file.")
        else:
            print(
                "\nNo functions matching the filter criteria were found or remained after filtering."
            )
        sys.exit(0)
    else:
        print(
            f"\nFiltered graph contains {len(call_graph.nodes)} functions and {len(call_graph.edges)} calls."
        )
    print("Detecting cycles in filtered graph...")
    cycles_list, cyclic_nodes, cyclic_edges = detect_cycles(call_graph)
    num_cycles = len(cycles_list)
    if cyclic_nodes:
        sorted_mangled_cyclic_nodes = sorted(list(cyclic_nodes))
        print(f"\nDetected {num_cycles} simple cycle(s) involving {len(sorted_mangled_cyclic_nodes)} function(s):")
        for mangled_node in sorted_mangled_cyclic_nodes:
            display_name = display_labels.get(mangled_node, mangled_node)
            print(f"  - {display_name} ({mangled_node})")
    else:
        print("\nNo cycles detected in the filtered call graph.")

    svg_path = args.svg_file
    if not svg_path:
        base_name = args.llvm_file
        if base_name.lower().endswith(".ll"):
            base_name = base_name[:-3]
        svg_path = base_name + ".filtered.svg"

    print(f"\nGenerating filtered graph visualization...")
    draw_and_export_graph(
        call_graph,
        display_labels,
        svg_path,
        cycles_list,
        assembly_code,
        cyclic_nodes=cyclic_nodes,
        cyclic_edges=cyclic_edges,
    )


if __name__ == "__main__":
    main()
