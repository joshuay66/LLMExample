// =============================================================================
// Program.cs - Educational LLM Training Pipeline
// =============================================================================
// This program demonstrates the complete lifecycle of a Large Language Model:
//
//   1. DATA LOADING: Read text files and convert them to tokens
//   2. MODEL CREATION: Build a GPT-2 transformer from scratch
//   3. TRAINING: Teach the model to predict the next token
//   4. INFERENCE: Use the trained model to generate new text
//   5. EXPORT: Save in a format compatible with LMStudio (via GGUF conversion)
//
// WHAT YOU'LL LEARN:
// ==================
// By reading through this code (start here, then follow the imports), you'll
// understand the fundamental building blocks of all modern LLMs:
//   - Tokenization: How text becomes numbers
//   - Embeddings: How numbers become meaningful vectors
//   - Self-Attention: How the model learns relationships between words
//   - Feed-Forward Networks: How the model processes information
//   - Training: How the model improves through gradient descent
//   - Generation: How the model produces new text
//
// HOW TO USE:
// ===========
// 1. Place your .txt training files in the data/training/ directory
// 2. Run: dotnet run --project src/LLMExample.Training
// 3. Watch the training progress (loss should decrease over time)
// 4. After training, run the Python conversion script for LMStudio:
//    python convert/convert_to_gguf.py exported_model model.gguf
//
// EXPECTED OUTPUT:
// ================
// The generated text will NOT be coherent (this is a tiny model with limited data).
// What matters is that the loss decreases during training, showing the model IS
// learning patterns from your text. For reference:
//   - Random guessing: loss ≈ 10.82 (log of vocabulary size)
//   - After training: loss should drop to 3-5 range
// =============================================================================

using System.Reflection;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;
using LLMExample.Training.Config;
using LLMExample.Training.Data;
using LLMExample.Training.Model;
using LLMExample.Training.Training;
using LLMExample.Training.Inference;
using LLMExample.Training.Export;

// =============================================================================
// NATIVE LIBRARY LOADING
// =============================================================================
// TorchSharp's built-in native library loader looks for .dylib files next to
// TorchSharp.dll, but .NET's NuGet restore places them in runtimes/{rid}/native/.
// We must explicitly load the native libraries in dependency order BEFORE
// TorchSharp's static constructor runs.
//
// The dependency chain is:
//   libomp → libc10 → libtorch_cpu → libtorch → libLibTorchSharp
// Each library depends on the ones before it being loaded.
{
    var assemblyDir = Path.GetDirectoryName(typeof(Program).Assembly.Location)!;
    var rid = RuntimeInformation.RuntimeIdentifier; // e.g., "osx-arm64"
    var nativeDir = Path.Combine(assemblyDir, "runtimes", rid, "native");

    if (Directory.Exists(nativeDir))
    {
        // Load native libraries in dependency order.
        // Each .dylib depends on the previous ones being already in memory.
        string[] libsInOrder = {
            "libomp.dylib",              // OpenMP runtime (parallel computation)
            "libc10.dylib",              // Core PyTorch tensor library
            "libtorch_cpu.dylib",        // CPU backend for PyTorch
            "libtorch.dylib",            // Main PyTorch library
            "libLibTorchSharp.dylib",    // C# ↔ LibTorch interop bridge
        };

        foreach (var lib in libsInOrder)
        {
            var libPath = Path.Combine(nativeDir, lib);
            if (File.Exists(libPath))
            {
                NativeLibrary.Load(libPath);
            }
        }
    }
    else
    {
        Console.WriteLine($"WARNING: Native library directory not found: {nativeDir}");
        Console.WriteLine("TorchSharp may fail to initialize. Try running 'dotnet restore'.");
    }
}

// =============================================================================
// STEP 1: Select the best available computing device
// =============================================================================
// Modern ML frameworks can run on different hardware:
//   - CUDA: NVIDIA GPUs (fastest for training)
//   - MPS: Apple Silicon GPUs (Metal Performance Shaders)
//   - CPU: Works everywhere but is slowest
//
// We try each in order of speed, falling back as needed.
Console.WriteLine("=== Educational LLM Training Pipeline ===\n");

// NOTE ON MPS (Apple Silicon GPU):
// While MPS is available on Apple Silicon, TorchSharp's MPS support has memory
// management issues with large vocabulary models — the backward pass during
// training can exhaust GPU memory even for our small model because the 50,257-
// token vocabulary creates very large gradient tensors. CPU on Apple Silicon
// is still fast (thanks to the unified memory architecture) and avoids these
// issues entirely. We use CPU by default for reliability.
Device device;
if (torch.cuda.is_available())
{
    device = torch.CUDA;
    Console.WriteLine("Using device: CUDA (NVIDIA GPU)");
}
else
{
    device = torch.CPU;
    Console.WriteLine("Using device: CPU");
    if (torch.mps_is_available())
    {
        Console.WriteLine("  (MPS is available but CPU is used for training stability)");
    }
}

// =============================================================================
// STEP 2: Configure the model
// =============================================================================
// All hyperparameters are defined in ModelConfig. You can experiment by
// changing these values! For example:
//   - Increase NLayer to 6 or 8 for a deeper model (slower but potentially better)
//   - Increase NEmbedding to 256 for more capacity (uses more memory)
//   - Decrease BlockSize to 128 if you're running out of memory
//   - Increase MaxSteps for longer training (better results)
var config = new ModelConfig
{
    // You can override defaults here, e.g.:
    // LearningRate = 1e-3f,
    MaxSteps = 500,  // Set lower for faster iteration; increase for better results
    BatchSize = 4,   // Smaller batch size for CPU training; increase if you have GPU memory
    LogEveryNSteps = 50,
};

Console.WriteLine($"\nModel configuration:");
Console.WriteLine($"  Embedding dim: {config.NEmbedding}");
Console.WriteLine($"  Attention heads: {config.NHead}");
Console.WriteLine($"  Transformer layers: {config.NLayer}");
Console.WriteLine($"  Context window: {config.BlockSize} tokens");
Console.WriteLine($"  FFN hidden dim: {config.NInner}");
Console.WriteLine($"  Vocabulary size: {config.VocabSize:N0}");

// =============================================================================
// STEP 3: Initialize the tokenizer
// =============================================================================
// The tokenizer converts text ↔ numbers. We use the standard GPT-2 BPE
// (Byte Pair Encoding) tokenizer with 50,257 tokens.
Console.WriteLine($"\nInitializing GPT-2 BPE tokenizer...");
var tokenizer = new TokenizerWrapper();

// Quick demo of tokenization:
var demoText = "Hello, world! This is a language model.";
var demoTokens = tokenizer.Encode(demoText);
Console.WriteLine($"  Example: \"{demoText}\"");
Console.WriteLine($"  Tokens:  [{string.Join(", ", demoTokens)}] ({demoTokens.Count} tokens)");
Console.WriteLine($"  Decoded: \"{tokenizer.Decode(demoTokens)}\"");

// =============================================================================
// STEP 4: Load and tokenize training data
// =============================================================================
// Load all .txt files from the data directory. Each file is tokenized and
// concatenated into one long stream of tokens, separated by EOS tokens.
Console.WriteLine($"\nLoading training data from '{config.DataDirectory}'...");
var dataset = new TextDataset();
dataset.Load(config.DataDirectory, tokenizer, config.BlockSize);

// =============================================================================
// STEP 5: Create the model
// =============================================================================
// Build the GPT-2 model and move it to the selected device.
// This allocates memory for all ~7.3M parameters.
Console.WriteLine($"\nCreating GPT-2 model...");
var model = new Gpt2Model(config);
model = model.to(device) as Gpt2Model ?? throw new InvalidOperationException("Failed to move model to device");

// Count and display total parameters
long totalParams = 0;
foreach (var param in model.parameters())
{
    totalParams += param.numel();
}
Console.WriteLine($"  Total parameters: {totalParams:N0} ({totalParams / 1_000_000.0:F1}M)");
Console.WriteLine($"  Model memory (float32): ~{totalParams * 4 / 1024.0 / 1024.0:F1} MB");

// Print the model's tensor names (useful for debugging export issues)
Console.WriteLine($"\n  Parameter names (for GGUF compatibility verification):");
foreach (var (name, param) in model.named_parameters())
{
    Console.WriteLine($"    {name} → {string.Join("x", param.shape)}");
}

// =============================================================================
// STEP 6: Train the model
// =============================================================================
// This is where the magic happens! The training loop repeatedly:
//   1. Samples random text chunks from the dataset
//   2. Asks the model to predict the next token at each position
//   3. Measures how wrong the predictions are (cross-entropy loss)
//   4. Adjusts the model's weights to improve predictions
var trainer = new Trainer(model, dataset, config, device);
trainer.Train();

// =============================================================================
// STEP 7: Generate sample text
// =============================================================================
// Now let's see what the model has learned! We feed it a prompt and let it
// generate new text one token at a time.
Console.WriteLine("\n=== Sample Text Generation ===\n");
var generator = new TextGenerator(model, tokenizer, config, device);

// Try a few different prompts
string[] prompts = { "The ", "Once upon a time", "In the " };
foreach (var prompt in prompts)
{
    Console.WriteLine($"Prompt: \"{prompt}\"");
    var generated = generator.Generate(prompt, maxNewTokens: 100, temperature: 0.8f);
    Console.WriteLine($"Generated: {generated}");
    Console.WriteLine();
}

// =============================================================================
// STEP 8: Export to HuggingFace format
// =============================================================================
// Export the model in a format that llama.cpp's converter can process.
// This creates a directory with the model weights, config, and tokenizer files.
var exporter = new HuggingFaceExporter();
exporter.Export(model, config, config.ExportDirectory, config.TokenizerDirectory);

// =============================================================================
// DONE! Next steps for GGUF conversion
// =============================================================================
Console.WriteLine("\n=== Next Steps: Convert to GGUF for LMStudio ===\n");
Console.WriteLine("1. Clone llama.cpp (if you haven't already):");
Console.WriteLine("   git clone https://github.com/ggml-org/llama.cpp.git\n");
Console.WriteLine("2. Create a Python virtual environment:");
Console.WriteLine("   python3 -m venv .venv");
Console.WriteLine("   source .venv/bin/activate\n");
Console.WriteLine("3. Install Python dependencies:");
Console.WriteLine("   pip install -r convert/requirements.txt\n");
Console.WriteLine("4. Run the conversion:");
Console.WriteLine($"   python convert/convert_to_gguf.py {config.ExportDirectory} model.gguf\n");
Console.WriteLine("5. Import into LMStudio:");
Console.WriteLine("   Copy model.gguf to ~/.cache/lm-studio/models/local/my-tiny-gpt2/");
Console.WriteLine("   Or use: lms import model.gguf\n");
Console.WriteLine("Note: This tiny model (~7.3M params) will generate text but it won't be");
Console.WriteLine("coherent like ChatGPT. The purpose is to understand the MECHANISM, not");
Console.WriteLine("to compete with production models that have 1000x more parameters.\n");
Console.WriteLine("=== Pipeline Complete! ===");
