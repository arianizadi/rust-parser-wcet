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

assembly_costs = {
    # --- Data Movement Instructions ---
    "movb": 1,  # Move byte (reg/imm/mem*)
    "movw": 1,  # Move word (reg/imm/mem*)
    "movl": 1,  # Move doubleword (reg/imm/mem*)
    "movq": 1,  # Move quadword (reg/imm/mem*) - *Memory ops depend heavily on cache (~3-5+ cycles L1 hit)
    "movabsq": 1,  # Move 64-bit immediate to register
    "movzbw": 1,  # Move byte to word with zero-extend
    "movzbl": 1,  # Move byte to doubleword with zero-extend
    "movzbq": 1,  # Move byte to quadword with zero-extend
    "movzwl": 1,  # Move word to doubleword with zero-extend
    "movzwq": 1,  # Move word to quadword with zero-extend
    "movzlq": 1,  # Move doubleword to quadword with zero-extend (implicit in movl to 64-bit reg)
    "movsbw": 1,  # Move byte to word with sign-extend
    "movsbl": 1,  # Move byte to doubleword with sign-extend
    "movsbq": 1,  # Move byte to quadword with sign-extend
    "movswl": 1,  # Move word to doubleword with sign-extend
    "movswq": 1,  # Move word to quadword with sign-extend
    "movslq": 1,  # Move doubleword to quadword with sign-extend
    "leaq": 1,  # Load Effective Address (address calculation, often used for arithmetic)
    "cmove": 2,  # Conditional move if equal (example CMOVcc) - Other CMOVcc are similar
    "cmovne": 2,  # Conditional move if not equal
    "cmovl": 2,  # Conditional move if less
    "cmovg": 2,  # Conditional move if greater
    "cmovle": 2,  # Conditional move if less or equal
    "cmovge": 2,  # Conditional move if greater or equal
    "cmovb": 2,  # Conditional move if below (unsigned)
    "cmova": 2,  # Conditional move if above (unsigned)
    "cmovbe": 2,  # Conditional move if below or equal (unsigned)
    "cmovae": 2,  # Conditional move if above or equal (unsigned)
    "cmovs": 2,  # Conditional move if sign
    "cmovns": 2,  # Conditional move if not sign
    "cmovp": 2,  # Conditional move if parity
    "cmovnp": 2,  # Conditional move if not parity
    "cmovo": 2,  # Conditional move if overflow
    "cmovno": 2,  # Conditional move if not overflow
    "xchgb": 2,  # Exchange byte (reg-reg)
    "xchgw": 2,  # Exchange word (reg-reg)
    "xchgl": 2,  # Exchange doubleword (reg-reg)
    "xchgq": 2,  # Exchange quadword (reg-reg) - Memory variant is much slower (atomic)
    # --- Stack Instructions ---
    "pushq": 2,  # Push quadword onto stack
    "popq": 2,  # Pop quadword from stack
    "pushfq": 5,  # Push RFLAGS register
    "popfq": 5,  # Pop RFLAGS register
    "enter": 10,  # Create stack frame (often slower than manual setup)
    "leave": 2,  # Destroy stack frame (mov rsp, rbp; pop rbp)
    # --- Integer Arithmetic Instructions ---
    "addb": 1,  # Add byte
    "addw": 1,  # Add word
    "addl": 1,  # Add doubleword
    "addq": 1,  # Add quadword
    "subb": 1,  # Subtract byte
    "subw": 1,  # Subtract word
    "subl": 1,  # Subtract doubleword
    "subq": 1,  # Subtract quadword
    "incb": 1,  # Increment byte
    "incw": 1,  # Increment word
    "incl": 1,  # Increment doubleword
    "incq": 1,  # Increment quadword
    "decb": 1,  # Decrement byte
    "decw": 1,  # Decrement word
    "decl": 1,  # Decrement doubleword
    "decq": 1,  # Decrement quadword
    "negb": 1,  # Negate byte
    "negw": 1,  # Negate word
    "negl": 1,  # Negate doubleword
    "negq": 1,  # Negate quadword
    "imulb": 3,  # Signed multiply (implicit AX = AL * src)
    "imulw": 3,  # Signed multiply (implicit DX:AX = AX * src)
    "imull": 3,  # Signed multiply (implicit EDX:EAX = EAX * src)
    "imulq": 3,  # Signed multiply (implicit RDX:RAX = RAX * src) - 2/3 operand forms also ~3 cycles latency
    "mulb": 3,  # Unsigned multiply (implicit AX = AL * src)
    "mulw": 3,  # Unsigned multiply (implicit DX:AX = AX * src)
    "mull": 3,  # Unsigned multiply (implicit EDX:EAX = EAX * src)
    "mulq": 3,  # Unsigned multiply (implicit RDX:RAX = RAX * src)
    "idivb": 20,  # Signed divide (implicit AL=AX/src, AH=rem) - **SLOW/VARIABLE**
    "idivw": 25,  # Signed divide (implicit AX=DX:AX/src, DX=rem) - **SLOW/VARIABLE**
    "idivl": 30,  # Signed divide (implicit EAX=EDX:EAX/src, EDX=rem) - **SLOW/VARIABLE**
    "idivq": 40,  # Signed divide (implicit RAX=RDX:RAX/src, RDX=rem) - **VERY SLOW/VARIABLE** (~20-80+)
    "divb": 20,  # Unsigned divide (implicit AL=AX/src, AH=rem) - **SLOW/VARIABLE**
    "divw": 25,  # Unsigned divide (implicit AX=DX:AX/src, DX=rem) - **SLOW/VARIABLE**
    "divl": 30,  # Unsigned divide (implicit EAX=EDX:EAX/src, EDX=rem) - **SLOW/VARIABLE**
    "divq": 40,  # Unsigned divide (implicit RAX=RDX:RAX/src, RDX=rem) - **VERY SLOW/VARIABLE** (~20-80+)
    "cbw": 1,  # Convert byte to word (sign extend AL->AX)
    "cwde": 1,  # Convert word to doubleword (sign extend AX->EAX)
    "cdqe": 1,  # Convert doubleword to quadword (sign extend EAX->RAX)
    "cwd": 1,  # Convert word to doubleword (sign extend AX->DX:AX for idivw)
    "cdq": 1,  # Convert doubleword to quadword (sign extend EAX->EDX:EAX for idivl)
    "cqo": 1,  # Convert quadword to octoword (sign extend RAX->RDX:RAX for idivq)
    # --- Logic and Bitwise Instructions ---
    "andb": 1,  # Bitwise AND byte
    "andw": 1,  # Bitwise AND word
    "andl": 1,  # Bitwise AND doubleword
    "andq": 1,  # Bitwise AND quadword
    "orb": 1,  # Bitwise OR byte
    "orw": 1,  # Bitwise OR word
    "orl": 1,  # Bitwise OR doubleword
    "orq": 1,  # Bitwise OR quadword
    "xorb": 1,  # Bitwise XOR byte
    "xorw": 1,  # Bitwise XOR word
    "xorl": 1,  # Bitwise XOR doubleword
    "xorq": 1,  # Bitwise XOR quadword
    "notb": 1,  # Bitwise NOT byte
    "notw": 1,  # Bitwise NOT word
    "notl": 1,  # Bitwise NOT doubleword
    "notq": 1,  # Bitwise NOT quadword
    "shlb": 1,  # Shift Left byte
    "shlw": 1,  # Shift Left word
    "shll": 1,  # Shift Left doubleword
    "shlq": 1,  # Shift Left quadword (SAL is same opcode)
    "shrb": 1,  # Logical Shift Right byte
    "shrw": 1,  # Logical Shift Right word
    "shrl": 1,  # Logical Shift Right doubleword
    "shrq": 1,  # Logical Shift Right quadword
    "sarb": 1,  # Arithmetic Shift Right byte
    "sarw": 1,  # Arithmetic Shift Right word
    "sarl": 1,  # Arithmetic Shift Right doubleword
    "sarq": 1,  # Arithmetic Shift Right quadword
    "rolb": 1,  # Rotate Left byte
    "rolw": 1,  # Rotate Left word
    "roll": 1,  # Rotate Left doubleword
    "rolq": 1,  # Rotate Left quadword
    "rorb": 1,  # Rotate Right byte
    "rorw": 1,  # Rotate Right word
    "rorl": 1,  # Rotate Right doubleword
    "rorq": 1,  # Rotate Right quadword
    "rclb": 2,  # Rotate Carry Left byte
    "rclw": 2,  # Rotate Carry Left word
    "rcll": 2,  # Rotate Carry Left doubleword
    "rclq": 2,  # Rotate Carry Left quadword
    "rcrb": 2,  # Rotate Carry Right byte
    "rcrw": 2,  # Rotate Carry Right word
    "rcrl": 2,  # Rotate Carry Right doubleword
    "rcrq": 2,  # Rotate Carry Right quadword
    "testb": 1,  # Logical AND, sets flags
    "testw": 1,  # Logical AND, sets flags
    "testl": 1,  # Logical AND, sets flags
    "testq": 1,  # Logical AND, sets flags
    "cmpb": 1,  # Compare bytes, sets flags
    "cmpw": 1,  # Compare words, sets flags
    "cmpl": 1,  # Compare doublewords, sets flags
    "cmpq": 1,  # Compare quadwords, sets flags
    "sete": 1,  # Set byte if equal (ZF=1) (Example SETcc) - others similar cost
    "setne": 1,  # Set byte if not equal (ZF=0)
    "setl": 1,  # Set byte if less (SF!=OF)
    "setg": 1,  # Set byte if greater (ZF=0 && SF==OF)
    "setle": 1,  # Set byte if less or equal (ZF=1 || SF!=OF)
    "setge": 1,  # Set byte if greater or equal (SF==OF)
    "setb": 1,  # Set byte if below (CF=1)
    "seta": 1,  # Set byte if above (CF=0 && ZF=0)
    "setbe": 1,  # Set byte if below or equal (CF=1 || ZF=1)
    "setae": 1,  # Set byte if above or equal (CF=0)
    "sets": 1,  # Set byte if sign (SF=1)
    "setns": 1,  # Set byte if not sign (SF=0)
    "setp": 1,  # Set byte if parity (PF=1)
    "setnp": 1,  # Set byte if not parity (PF=0)
    "seto": 1,  # Set byte if overflow (OF=1)
    "setno": 1,  # Set byte if not overflow (OF=0)
    "bt": 2,  # Bit Test
    "bts": 2,  # Bit Test and Set
    "btr": 2,  # Bit Test and Reset
    "btc": 2,  # Bit Test and Complement
    "bsf": 3,  # Bit Scan Forward
    "bsr": 3,  # Bit Scan Reverse
    "popcnt": 3,  # Population Count (count set bits) - modern CPUs
    "lzcnt": 3,  # Leading Zero Count
    "tzcnt": 3,  # Trailing Zero Count
    # --- Control Flow Instructions ---
    "jmp": 1,  # Unconditional Jump (cost assumes predicted; mispredict ~15-25+)
    "je": 1,  # Jump if equal (example Jcc) - cost assumes predicted; mispredict ~15-25+
    "jne": 1,  # Jump if not equal
    "jl": 1,  # Jump if less
    "jg": 1,  # Jump if greater
    "jle": 1,  # Jump if less or equal
    "jge": 1,  # Jump if greater or equal
    "jb": 1,  # Jump if below
    "ja": 1,  # Jump if above
    "jbe": 1,  # Jump if below or equal
    "jae": 1,  # Jump if above or equal
    "js": 1,  # Jump if sign
    "jns": 1,  # Jump if not sign
    "jp": 1,  # Jump if parity
    "jnp": 1,  # Jump if not parity
    "jo": 1,  # Jump if overflow
    "jno": 1,  # Jump if not overflow
    "callq": 5,  # Procedure Call (base cost + branch prediction effects)
    "retq": 3,  # Return from Procedure (base cost + branch prediction effects)
    "loop": 6,  # Decrement CX and loop if not zero
    "loope": 6,  # Decrement CX and loop if equal
    "loopne": 6,  # Decrement CX and loop if not equal
    # --- Floating Point & SIMD (SSE/AVX - Selected Scalar & Packed) ---
    "movss": 1,  # Move Scalar Single FP (reg/imm/mem*)
    "movsd": 1,  # Move Scalar Double FP (reg/imm/mem*)
    "movaps": 1,  # Move Aligned Packed Single FP (reg/mem*)
    "movapd": 1,  # Move Aligned Packed Double FP (reg/mem*)
    "movups": 2,  # Move Unaligned Packed Single FP (reg/mem*) - Slightly higher penalty if unaligned
    "movupd": 2,  # Move Unaligned Packed Double FP (reg/mem*) - Slightly higher penalty if unaligned
    "movdqa": 1,  # Move Aligned Double Quadword (128-bit integer)
    "movdqu": 2,  # Move Unaligned Double Quadword (128-bit integer)
    "addss": 4,  # Add Scalar Single FP
    "addsd": 4,  # Add Scalar Double FP
    "addps": 4,  # Add Packed Single FP
    "addpd": 4,  # Add Packed Double FP
    "subss": 4,  # Subtract Scalar Single FP
    "subsd": 4,  # Subtract Scalar Double FP
    "subps": 4,  # Subtract Packed Single FP
    "subpd": 4,  # Subtract Packed Double FP
    "mulss": 4,  # Multiply Scalar Single FP
    "mulsd": 4,  # Multiply Scalar Double FP
    "mulps": 4,  # Multiply Packed Single FP
    "mulpd": 4,  # Multiply Packed Double FP
    "divss": 15,  # Divide Scalar Single FP - **Slow/Variable**
    "divsd": 20,  # Divide Scalar Double FP - **Slow/Variable**
    "divps": 15,  # Divide Packed Single FP - **Slow/Variable**
    "divpd": 20,  # Divide Packed Double FP - **Slow/Variable**
    "sqrtss": 15,  # Square Root Scalar Single FP - **Slow/Variable**
    "sqrtsd": 20,  # Square Root Scalar Double FP - **Slow/Variable**
    "sqrtps": 15,  # Square Root Packed Single FP - **Slow/Variable**
    "sqrtpd": 20,  # Square Root Packed Double FP - **Slow/Variable**
    "ucomiss": 3,  # Unordered Compare Scalar Single FP (sets EFLAGS)
    "ucomisd": 3,  # Unordered Compare Scalar Double FP (sets EFLAGS)
    "cmpps": 3,  # Compare Packed Single FP (result in register/mask)
    "cmppd": 3,  # Compare Packed Double FP (result in register/mask)
    "cvtsi2ss": 5,  # Convert Int to Scalar Single FP
    "cvtsi2sd": 5,  # Convert Int to Scalar Double FP
    "cvttss2si": 5,  # Convert Truncated Scalar Single FP to Int
    "cvttsd2si": 5,  # Convert Truncated Scalar Double FP to Int
    "shufps": 2,  # Shuffle Packed Single FP
    "shufpd": 2,  # Shuffle Packed Double FP
    "paddb": 1,  # Packed Add Byte (MMX/SSE/AVX)
    "paddw": 1,  # Packed Add Word
    "paddd": 1,  # Packed Add Doubleword
    "paddq": 1,  # Packed Add Quadword
    "psubb": 1,  # Packed Subtract Byte
    "psubw": 1,  # Packed Subtract Word
    "psubd": 1,  # Packed Subtract Doubleword
    "psubq": 1,  # Packed Subtract Quadword
    "pand": 1,  # Packed Bitwise AND
    "por": 1,  # Packed Bitwise OR
    "pxor": 1,  # Packed Bitwise XOR
    # --- String Instructions ---
    # Base cost per element; REP prefix makes total cost highly variable
    "movsb": 8,  # Move byte from DS:[RSI] to ES:[RDI]
    "movsw": 8,  # Move word
    "movsl": 8,  # Move doubleword
    "movsq": 8,  # Move quadword
    "cmpsb": 8,  # Compare byte DS:[RSI] with ES:[RDI]
    "cmpsw": 8,  # Compare word
    "cmpsl": 8,  # Compare doubleword
    "cmpsq": 8,  # Compare quadword
    "stosb": 8,  # Store AL to ES:[RDI]
    "stosw": 8,  # Store AX
    "stosl": 8,  # Store EAX
    "stosq": 8,  # Store RAX
    "lodsb": 8,  # Load byte from DS:[RSI] to AL
    "lodsw": 8,  # Load word to AX
    "lodsl": 8,  # Load doubleword to EAX
    "lodsq": 8,  # Load quadword to RAX
    "scasb": 8,  # Scan AL against ES:[RDI]
    "scasw": 8,  # Scan AX
    "scasl": 8,  # Scan EAX
    "scasq": 8,  # Scan RAX
    # --- System and Miscellaneous Instructions ---
    "syscall": 1000,  # Fast System Call - **Very Slow (Kernel Transition)**
    "sysenter": 1000,  # Legacy System Call - **Very Slow**
    "sysexit": 1000,  # Legacy System Return - **Very Slow**
    "int": 1000,  # Software Interrupt - **Very Slow (Handler Dependent)**
    "cpuid": 100,  # CPU Identification - Slow, Serializing
    "rdtsc": 30,  # Read Time-Stamp Counter - Serializing effects
    "rdtscp": 30,  # Read Time-Stamp Counter and Processor ID - Serializing effects
    "nop": 1,  # No Operation (often 0 cycles effective)
    "pause": 40,  # Spin Loop Hint (variable, yields to hyperthread)
    "hlt": 1000,  # Halt - Stops CPU until interrupt (cost is context-dependent)
    "lock": 15,  # Atomic Prefix (Adds significant cost to the following instruction) - **Base Cost Added**
    "cmpxchgb": 15,  # Compare and Exchange byte (use with LOCK)
    "cmpxchgw": 15,  # Compare and Exchange word
    "cmpxchgl": 15,  # Compare and Exchange doubleword
    "cmpxchgq": 15,  # Compare and Exchange quadword
    "xaddb": 15,  # Exchange and Add byte (use with LOCK)
    "xaddw": 15,  # Exchange and Add word
    "xaddl": 15,  # Exchange and Add doubleword
    "xaddq": 15,  # Exchange and Add quadword
    "lfence": 20,  # Load Fence
    "sfence": 20,  # Store Fence
    "mfence": 30,  # Memory Fence (Load + Store)
}

def demangle_name(mangled_name):
    """
    Demangle a C++/Rust mangled symbol name, and clean up Rust hash suffixes and closure/generic markers.
    """
    if not mangled_name:
        return ""
    name_to_demangle = mangled_name.lstrip('@')
    try:
        demangled = cxxfilt.demangle(name_to_demangle)
        # Remove Rust hash suffixes (e.g., ::h1234567890abcdefE)
        demangled = re.sub(r'::h[0-9a-f]{16}E$', '', demangled)
        # Remove Rust closure markers
        demangled = re.sub(r"\{\{closure\}\}", "", demangled)
        # Remove Rust generator markers
        demangled = re.sub(r"\{\{generator\}\}", "", demangled)
        # Remove trailing whitespace
        demangled = demangled.strip()
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


def _is_rust_stdlib_or_core(label):
    # Returns True if the demangled label is from Rust std/core/alloc or LLVM intrinsics
    return (
        label.startswith("std::")
        or label.startswith("core::")
        or label.startswith("alloc::")
        or label.startswith("llvm.")
    )


def _is_rust_closure_or_generator(label):
    # Returns True if the demangled label contains closure or generator markers
    return "{{closure}}" in label or "{{generator}}" in label


def _is_rust_generic(label):
    # Returns True if the demangled label contains generic angle brackets (not for LLVM intrinsics)
    return ("<" in label and ">" in label) and not label.startswith("llvm.")


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
    print(
        "\nFiltering nodes (excluding std/core/alloc, LLVM intrinsics, closures, and generics):"
    )
    kept_nodes = 0
    filtered_out_nodes = 0
    for node, raw_label in full_labels.items():
        keep_node = True
        # Exclude Rust stdlib/core/alloc and LLVM intrinsics
        if _is_rust_stdlib_or_core(raw_label):
            keep_node = False
        # Exclude closures and generators
        elif _is_rust_closure_or_generator(raw_label):
            keep_node = False
        # Exclude generics (angle brackets) except for LLVM intrinsics
        elif _is_rust_generic(raw_label):
            keep_node = False
        # Exclude unmangled names that look like _ZN... (Rust/C++ mangling)
        elif raw_label.startswith("_ZN"):
            keep_node = False
        if keep_node:
            filtered_graph.add_node(node)
            simplified_label = raw_label
            # Remove Rust trait-as-impl prefix (e.g., <T as Trait>::)
            simplified_label = re.sub(r'<.* as .*>::', '', simplified_label)
            # Remove generic parameters for display clarity (keep only the function name)
            simplified_label = re.sub(r"<.*?>", "", simplified_label)
            # Remove extra whitespace
            simplified_label = simplified_label.strip()
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
    parser_arg.add_argument("asm_file", help="Path to the Assembly source file (.s).")
    parser_arg.add_argument("llvm_file", help="Path to the LLVM IR source file (.ll).")
    parser_arg.add_argument(
        "--no-filter",
        dest="no_filter",
        action="store_true",
        help="Disable all filtering of functions; include all nodes and edges in the call graph.",
    )
    parser_arg.add_argument(
        "--svg",
        dest="svg_file",
        default=None,
        help="Path to output SVG file (default: <llvm_file_base>.filtered.svg)",
    )
    args = parser_arg.parse_args()

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
