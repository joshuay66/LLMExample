// =============================================================================
// SafeTensorsWriter.cs - Writes the SafeTensors File Format
// =============================================================================
// SafeTensors is a simple, safe file format for storing tensors (multi-
// dimensional arrays of numbers). It was created by HuggingFace as a
// replacement for Python's pickle format (which has security vulnerabilities).
//
// FILE FORMAT:
// ============
// The file structure is straightforward:
//   [8 bytes]      Header size (little-endian uint64)
//   [N bytes]      JSON header with tensor metadata
//   [remaining]    Raw tensor data (contiguous bytes)
//
// The JSON header looks like:
//   {
//     "tensor_name": {
//       "dtype": "F32",
//       "shape": [50257, 128],
//       "data_offsets": [0, 25731072]
//     },
//     ...
//   }
//
// Each tensor's data is stored as raw bytes in row-major (C) order,
// starting at the offset specified in "data_offsets".
//
// WHY SAFETENSORS?
// ================
// - Simple: easy to implement from scratch (as we do here!)
// - Safe: no arbitrary code execution (unlike pickle)
// - Fast: memory-mappable, no deserialization overhead
// - Standard: used by HuggingFace, llama.cpp, and most ML tools
// =============================================================================

using System.Text;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMExample.Training.Export;

public static class SafeTensorsWriter
{
    /// <summary>
    /// Write a collection of named tensors to a safetensors file.
    /// All tensors are converted to float32 and stored on CPU.
    /// </summary>
    /// <param name="path">Output file path</param>
    /// <param name="tensors">Dictionary of tensor name → tensor data</param>
    public static void Write(string path, Dictionary<string, Tensor> tensors)
    {
        // =====================================================================
        // STEP 1: Prepare all tensors — ensure they're contiguous, on CPU, float32
        // =====================================================================
        var prepared = new Dictionary<string, Tensor>();
        foreach (var (name, tensor) in tensors)
        {
            // Move to CPU (if on GPU/MPS) and convert to float32
            // contiguous() ensures the data is laid out in memory without gaps
            prepared[name] = tensor.to(ScalarType.Float32).cpu().contiguous();
        }

        // =====================================================================
        // STEP 2: Calculate byte offsets for each tensor
        // =====================================================================
        // We need to know where each tensor's data starts and ends in the file.
        // Tensors are stored sequentially after the header.
        var metadata = new Dictionary<string, object>();
        long currentOffset = 0;

        // Sort tensor names for deterministic output
        var sortedNames = prepared.Keys.OrderBy(k => k).ToList();

        foreach (var name in sortedNames)
        {
            var tensor = prepared[name];
            long numBytes = tensor.numel() * sizeof(float); // 4 bytes per float32

            // Build metadata for this tensor
            metadata[name] = new
            {
                dtype = "F32",
                shape = tensor.shape,
                data_offsets = new long[] { currentOffset, currentOffset + numBytes }
            };

            currentOffset += numBytes;
        }

        // =====================================================================
        // STEP 3: Serialize the JSON header
        // =====================================================================
        // The header is a JSON object with an entry for each tensor.
        // We need to know its exact byte size to write the 8-byte prefix.
        var headerJson = JsonSerializer.Serialize(metadata, new JsonSerializerOptions
        {
            WriteIndented = false // Compact JSON to minimize file size
        });
        var headerBytes = Encoding.UTF8.GetBytes(headerJson);

        // Pad header to 8-byte alignment (required by the spec)
        int paddedHeaderSize = headerBytes.Length;
        if (paddedHeaderSize % 8 != 0)
        {
            paddedHeaderSize += 8 - (paddedHeaderSize % 8);
        }
        var paddedHeader = new byte[paddedHeaderSize];
        Array.Copy(headerBytes, paddedHeader, headerBytes.Length);
        // Fill padding with spaces (0x20)
        for (int i = headerBytes.Length; i < paddedHeaderSize; i++)
        {
            paddedHeader[i] = 0x20;
        }

        // =====================================================================
        // STEP 4: Write the file
        // =====================================================================
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);

        // Write 8-byte header size (little-endian uint64)
        writer.Write((ulong)paddedHeaderSize);

        // Write the padded JSON header
        writer.Write(paddedHeader);

        // Write tensor data sequentially
        foreach (var name in sortedNames)
        {
            var tensor = prepared[name];
            long numElements = tensor.numel();

            // Get the raw float data from the tensor
            var data = new float[numElements];
            tensor.data<float>().CopyTo(data, 0);

            // Convert to bytes and write
            var bytes = new byte[numElements * sizeof(float)];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            writer.Write(bytes);
        }

        // Dispose prepared tensors
        foreach (var tensor in prepared.Values)
        {
            tensor.Dispose();
        }

        Console.WriteLine($"  Saved {sortedNames.Count} tensors to {path}");
        Console.WriteLine($"  Header size: {paddedHeaderSize:N0} bytes");
        Console.WriteLine($"  Data size: {currentOffset:N0} bytes ({currentOffset / 1024.0 / 1024.0:F1} MB)");
    }
}
