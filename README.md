# T-REX (Rust WCET Analyzer)

## Overview

T-REX (Timing Retriever and EXplorer) is a Python tool designed to analyze LLVM Intermediate Representation (IR) code, typically generated from Rust programs (`rustc --emit=llvm-ir`). It focuses on estimating the Worst-Case Execution Time (WCET) for individual functions based on heuristic instruction costs, visualizing the filtered function call graph, and detecting cycles within that graph.

This tool helps in understanding the structure and potential timing characteristics of the low-level code generated by the compiler *before* actual hardware testing.

![Linear](linear.filtered.svg)

## Features

* **LLVM IR Parsing:** Parses `.ll` files to identify function definitions and call sites.
* **Call Graph Construction:** Builds a directed graph representing the calls between functions.
* **Name Demangling:** Uses `cxxfilt` to demangle C++/Rust function names for better readability.
* **Graph Filtering:** Filters the call graph to focus on user-defined functions, excluding standard library, core library, closures, and generic functions by default.
* **Cycle Detection:** Identifies and reports simple cycles (potential recursion or complex control flow) within the filtered graph.
* **Heuristic WCET Estimation:** Calculates an estimated WCET in abstract "cycles" for each function.
    * **Note:** This estimation is based on predefined heuristic costs for LLVM instructions found.
    * The estimation includes a basic heuristic to penalize potential loops identified via back-edges.
* **SVG Visualization:** Exports the filtered call graph to an SVG file using Matplotlib and NetworkX.
    * Highlights nodes and edges involved in cycles.
    * Displays the demangled function name and its estimated WCET below each node.

## Usage

Run the script from your terminal, providing the path to the LLVM IR file (`.ll`) as an argument.

```bash
./create_ir.sh linear.rs

python3 main.py linear.ll
------------ or ---------------------
python3 main.py linear.ll --no-filter
```
