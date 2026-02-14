#!/usr/bin/env python3
"""
convert_to_gguf.py - Convert HuggingFace GPT-2 model to GGUF format
=====================================================================
This script converts the exported HuggingFace-format model directory
into a GGUF file that LMStudio (and other llama.cpp-based tools) can load.

GGUF (GPT-Generated Unified Format) is the standard format used by llama.cpp.
It packages model weights, tokenizer, and metadata into a single binary file.

Prerequisites:
    1. Clone llama.cpp: git clone https://github.com/ggml-org/llama.cpp.git
    2. Install dependencies: pip install -r requirements.txt
    3. Set LLAMA_CPP_DIR environment variable (or place llama.cpp next to this project)

Usage:
    python convert_to_gguf.py <model_dir> [output.gguf]

Examples:
    python convert_to_gguf.py exported_model model.gguf
"""

import subprocess
import sys
import os


def find_llama_cpp():
    """Find the llama.cpp directory by checking common locations."""
    # Check environment variable first
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    # Check common relative locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "llama.cpp"),
        os.path.join(script_dir, "..", "..", "llama.cpp"),
        os.path.expanduser("~/llama.cpp"),
    ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)

    return None


def main():
    # Parse arguments
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "exported_model"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "model.gguf"

    # Validate model directory exists
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        print("Run the C# training program first to export the model.")
        sys.exit(1)

    # Check for required files
    required_files = ["config.json", "model.safetensors", "vocab.json", "merges.txt"]
    for f in required_files:
        path = os.path.join(model_dir, f)
        if not os.path.isfile(path):
            print(f"Error: Required file '{f}' not found in {model_dir}/")
            sys.exit(1)

    # Find llama.cpp
    llama_cpp_dir = find_llama_cpp()
    if llama_cpp_dir is None:
        print("Error: Could not find llama.cpp directory.")
        print("Please either:")
        print("  1. Clone it: git clone https://github.com/ggml-org/llama.cpp.git")
        print("  2. Set LLAMA_CPP_DIR environment variable")
        sys.exit(1)

    # Find the conversion script
    converter = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.isfile(converter):
        print(f"Error: convert_hf_to_gguf.py not found in {llama_cpp_dir}")
        print("Make sure you have the latest version of llama.cpp.")
        sys.exit(1)

    print(f"Model directory: {model_dir}")
    print(f"Output file: {output_file}")
    print(f"llama.cpp dir: {llama_cpp_dir}")
    print(f"Converter: {converter}")
    print()

    # Build the conversion command
    # --no-lazy: Workaround for a known GPT-2 buffer alignment bug
    #            (see https://github.com/ggml-org/llama.cpp/issues/16013)
    # --outtype f16: Use float16 for smaller file size (half the size of f32)
    #               The quality loss from f16 is negligible for this model size
    cmd = [
        sys.executable,
        converter,
        model_dir,
        "--outfile", output_file,
        "--outtype", "f16",
        "--no-lazy",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)
        print()
        print(f"Success! GGUF model saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
        print()
        print("To use in LMStudio:")
        print(f"  1. Copy {output_file} to ~/.cache/lm-studio/models/local/my-tiny-gpt2/")
        print("  2. Open LMStudio and select the model")
        print("  3. Start a chat (output will be basic - this is a tiny educational model)")
    except subprocess.CalledProcessError as e:
        print(f"\nConversion failed with exit code {e.returncode}")
        print("Common issues:")
        print("  - Missing Python dependencies: pip install -r requirements.txt")
        print("  - Tensor name mismatch: check that model.safetensors has correct names")
        print("  - Buffer alignment: make sure --no-lazy flag is included")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nError: Python not found at '{sys.executable}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
