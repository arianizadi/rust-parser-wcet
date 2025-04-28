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


def estimate_wcet_cycles(function_name, llvm_code):
    costs = {
        # Memory operations
        "load": 100,  # Load from memory
        "store": 100,  # Store to memory
        "atomicrmw": 150,  # Atomic read-modify-write
        "cmpxchg": 200,  # Atomic compare and exchange
        "fence": 10,  # Memory fence/barrier
        # Stack and pointer operations
        "alloca": 5,  # Stack allocation
        "getelementptr": 2,  # Pointer arithmetic
        # Integer arithmetic
        "add": 1,  # Integer addition
        "sub": 1,  # Integer subtraction
        "mul": 5,  # Integer multiplication
        "sdiv": 40,  # Signed integer division
        "udiv": 40,  # Unsigned integer division
        "srem": 40,  # Signed integer remainder
        "urem": 40,  # Unsigned integer remainder
        # Floating point arithmetic
        "fadd": 4,  # Floating-point addition
        "fsub": 4,  # Floating-point subtraction
        "fmul": 7,  # Floating-point multiplication
        "fdiv": 40,  # Floating-point division
        "frem": 40,  # Floating-point remainder
        # Bitwise operations
        "shl": 1,  # Shift left
        "lshr": 1,  # Logical shift right
        "ashr": 1,  # Arithmetic shift right
        "and": 1,  # Bitwise AND
        "or": 1,  # Bitwise OR
        "xor": 1,  # Bitwise XOR
        "not": 1,  # Bitwise NOT (not an LLVM instruction, but included for completeness)
        # Comparisons
        "icmp": 1,  # Integer comparison
        "fcmp": 2,  # Floating-point comparison
        # Control flow
        "br": 3,  # Branch
        "switch": 10,  # Switch statement
        "indirectbr": 15,  # Indirect branch
        "call": 40,  # Function call
        "invoke": 50,  # Exception-aware function call
        "ret": 3,  # Return
        # PHI and selection
        "phi": 1,  # PHI node (SSA merge)
        "select": 2,  # Select/ternary operator
        # Vector operations
        "extractelement": 2,  # Extract element from vector
        "insertelement": 2,  # Insert element into vector
        "shufflevector": 4,  # Shuffle vector elements
        # Type conversions
        "zext": 1,  # Zero extension
        "sext": 1,  # Sign extension
        "trunc": 1,  # Truncate
        "fptrunc": 1,  # FP truncate
        "fpext": 1,  # FP extend
        "fptoui": 2,  # FP to unsigned int
        "fptosi": 2,  # FP to signed int
        "uitofp": 2,  # Unsigned int to FP
        "sitofp": 2,  # Signed int to FP
        "ptrtoint": 1,  # Pointer to int
        "inttoptr": 1,  # Int to pointer
        "bitcast": 1,  # Bitcast
        # Memory intrinsics
        "memcpy": 200,  # Memory copy
        "memmove": 200,  # Memory move
        "memset": 150,  # Memory set
        # Exception handling
        "landingpad": 5,  # Landing pad for exceptions
        "resume": 5,  # Resume exception propagation
        "unwind": 0,  # Unwind (obsolete in LLVM, but included for completeness)
        # More: Vector math and advanced instructions
        "va_arg": 3,  # Variable argument handling
        "extractvalue": 2,  # Extract value from aggregate
        "insertvalue": 2,  # Insert value into aggregate
        "freeze": 1,  # Freeze undefined value
        # Atomic instructions
        "atomicload": 120,  # Atomic load (not a direct LLVM instruction, but for modeling)
        "atomicstore": 120,  # Atomic store (not a direct LLVM instruction, but for modeling)
        # Debug and metadata (no cost)
        "dbg": 0,  # Debug info
        "llvm.dbg": 0,  # LLVM debug intrinsic
        # Barrier and synchronization
        "barrier": 20,  # Synchronization barrier (for parallel code)
        # Miscellaneous
        "unreachable": 0,  # Unreachable code
        "nop": 0,  # No operation (not an LLVM instruction, but for completeness)
        "assume": 0,  # llvm.assume intrinsic
        "lifetime.start": 0,  # llvm.lifetime.start intrinsic
        "lifetime.end": 0,  # llvm.lifetime.end intrinsic
        # Floating-point math intrinsics
        "llvm.sqrt": 30,  # Square root
        "llvm.pow": 50,  # Power
        "llvm.exp": 40,  # Exponential
        "llvm.log": 40,  # Logarithm
        "llvm.sin": 40,  # Sine
        "llvm.cos": 40,  # Cosine
        "llvm.floor": 10,  # Floor
        "llvm.ceil": 10,  # Ceil
        "llvm.round": 10,  # Round
        # Saturating arithmetic (LLVM 16+)
        "sadd_sat": 2,  # Signed saturating add
        "uadd_sat": 2,  # Unsigned saturating add
        "ssub_sat": 2,  # Signed saturating sub
        "usub_sat": 2,  # Unsigned saturating sub
        # Bit manipulation
        "ctpop": 5,  # Count population (number of set bits)
        "ctlz": 5,  # Count leading zeros
        "cttz": 5,  # Count trailing zeros
        # Overflow intrinsics
        "sadd_with_overflow": 2,
        "uadd_with_overflow": 2,
        "ssub_with_overflow": 2,
        "usub_with_overflow": 2,
        "smul_with_overflow": 7,
        "umul_with_overflow": 7,
        # Vector reductions
        "vector.reduce.add": 3,
        "vector.reduce.mul": 5,
        "vector.reduce.and": 2,
        "vector.reduce.or": 2,
        "vector.reduce.xor": 2,
        "vector.reduce.smax": 3,
        "vector.reduce.smin": 3,
        "vector.reduce.umax": 3,
        "vector.reduce.umin": 3,
    }
    try:
        lines = llvm_code.splitlines()
        function_body_lines = []
        in_function = False
        found_function = False
        target_func_name = function_name.lstrip("@")
        define_pattern = re.compile(
            r"^\s*define\s+.*?\s+@" + re.escape(target_func_name) + r"\s*\("
        )
        for line in lines:
            stripped_line = line.strip()
            if not in_function:
                if define_pattern.match(stripped_line):
                    in_function = True
                    found_function = True
                    if '{' in stripped_line:
                        body_part = stripped_line.split("{", 1)[1]
                        if re.match(r"^\s*\}\s*(;.*)?$", body_part):
                            single_line_body = body_part.rsplit("}", 1)[0]
                            if single_line_body.strip():
                                function_body_lines.append(single_line_body)
                            in_function = False
                        elif body_part.strip():
                            function_body_lines.append(body_part)
                    continue
            else:
                if re.match(r"^\s*\}\s*(;.*)?$", stripped_line):
                    in_function = False
                    break
                else:
                    function_body_lines.append(line)
        if not found_function:
            declare_pattern = re.compile(r"^\s*declare\s+.*?\s+@" + re.escape(target_func_name) + r"\s*\(")
            is_declaration = any(declare_pattern.match(l.strip()) for l in lines)
            if is_declaration:
                return 1
            else:
                print(
                    f"Warning: Could not find definition for {function_name}",
                    file=sys.stderr,
                )
                return None
        if in_function:
            print(
                f"Warning: Reached end of file while still inside function {function_name} (missing '}}'?)",
                file=sys.stderr,
            )
            return None
        function_body = "\n".join(function_body_lines)
        if not function_body.strip() and found_function:
            return 5
        estimated_cycles = 0
        instruction_count = 0
        body_lines_for_analysis = function_body.splitlines()
        for line in body_lines_for_analysis:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith(';') or stripped_line.endswith(':'):
                continue
            match_opcode = re.match(r'\s*(?:%[\w.]+\s*=\s*)?(\w+)', stripped_line)
            if match_opcode:
                opcode = match_opcode.group(1)
                if opcode in costs:
                    estimated_cycles += costs[opcode]
                    instruction_count += 1
        basic_blocks = len(re.findall(r'^\s*[\w.]+:', function_body, re.MULTILINE)) + 1
        estimated_cycles += basic_blocks * 1
        min_cost = 5 + instruction_count
        estimated_cycles = max(estimated_cycles, min_cost)
        labels = {}
        back_edges = 0
        for i, line in enumerate(body_lines_for_analysis):
            label_match = re.match(r"^\s*([\w.]+):", line.strip())
            if label_match:
                labels[label_match.group(1)] = i
        for i, line in enumerate(body_lines_for_analysis):
            branch_match = re.search(r'\bbr\s+label\s+%([\w.]+)', line)
            if branch_match:
                target_label = branch_match.group(1)
                if target_label in labels and labels[target_label] < i:
                    back_edges += 1
        if back_edges > 0:
            loop_penalty = back_edges * (estimated_cycles * 0.5)
            estimated_cycles += int(loop_penalty)
        return estimated_cycles
    except Exception as e:
        print(f"Error estimating cycles for {function_name}: {e}", file=sys.stderr)
        demangled_name = demangle_name(function_name)
        print(f"  - {demangled_name}: Unable to estimate cycles", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def draw_and_export_graph(
    graph,
    labels,
    svg_path,
    cycles_list,
    llvm_code,
    cyclic_nodes=None,
    cyclic_edges=None,
):
    if not graph.nodes:
        print("Filtered graph is empty, cannot draw.", file=sys.stderr)
        return
    edge_cycle_counts = calculate_edge_cycle_counts(graph, cycles_list)
    wcet_estimates = {}
    print(
        "Estimating WCET cycles for each function (independent, not including callees)..."
    )
    for node, display_label in labels.items():
        wcet_cycles = estimate_wcet_cycles(node, llvm_code)
        if wcet_cycles is not None:
            wcet_estimates[node] = wcet_cycles
            print(f"  - {display_label}: ~{wcet_cycles} cycles")
        else:
            print(f"  - {display_label}: WCET estimation failed")
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
                label_text = f"CPU Cycles: ~{wcet}"
            else:
                label_text = "CPU Cycles: N/A"
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
        plt.title(f"Filtered LLVM IR Call Graph (Cycles Highlighted)\nSource: {file_base_name}.ll", fontsize=16)
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
        description="Analyze LLVM IR code (.ll), filter graph to user functions, detect cycles, demangle names, estimate WCET, and export as SVG."
    )
    parser_arg.add_argument("llvm_file", help="Path to the LLVM IR source file (.ll).")
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
        llvm_code,
        cyclic_nodes=cyclic_nodes,
        cyclic_edges=cyclic_edges,
    )


if __name__ == "__main__":
    main()
