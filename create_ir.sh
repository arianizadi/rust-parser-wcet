#!/bin/bash

# Check if a Rust file was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <rust_file.rs>"
    exit 1
fi

# Get the input file name
INPUT_FILE=$1

# Check if the file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

# Extract the base name without extension
BASE_NAME=$(basename "$INPUT_FILE" .rs)

# Create LLVM IR
echo "Generating LLVM IR for $INPUT_FILE..."
rustc --emit=llvm-ir "$INPUT_FILE" -o "$BASE_NAME.ll"

if [ -f "$BASE_NAME.ll" ]; then
    echo "Successfully created $BASE_NAME.ll"
else
    echo "Failed to generate LLVM IR"
    exit 1
fi

echo "Done."
