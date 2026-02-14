// =============================================================================
// HuggingFaceExporter.cs - Export Model to HuggingFace Format
// =============================================================================
// To convert our trained model to GGUF format (for LMStudio), we first need
// to export it in a format that the llama.cpp conversion tool understands.
// That format is the HuggingFace model directory structure:
//
//   exported_model/
//     config.json              ← Model architecture description
//     model.safetensors        ← Trained weights in safetensors format
//     vocab.json               ← BPE vocabulary mapping
//     merges.txt               ← BPE merge rules
//     tokenizer_config.json    ← Tokenizer settings
//     special_tokens_map.json  ← Special token definitions
//
// CRITICAL: TENSOR NAME MAPPING
// ==============================
// The llama.cpp GGUF converter expects tensor names to match HuggingFace's
// GPT-2 naming convention EXACTLY. Our TorchSharp model uses the same names
// thanks to the nested module pattern, but we need to verify this.
//
// CRITICAL: Conv1D WEIGHT TRANSPOSITION
// =====================================
// HuggingFace's original GPT-2 implementation uses Conv1D layers (NOT standard
// Linear layers) for the attention and FFN projections. Conv1D stores weights
// transposed compared to Linear:
//   - Linear (our model): weight shape is [out_features, in_features]
//   - Conv1D (HuggingFace): weight shape is [in_features, out_features]
//
// The llama.cpp converter EXPECTS Conv1D format, so we must transpose
// the weights of c_attn, c_proj, and c_fc during export.
// =============================================================================

using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;
using LLMExample.Training.Config;
using LLMExample.Training.Model;

namespace LLMExample.Training.Export;

public class HuggingFaceExporter
{
    // These are the layer names where GPT-2 uses Conv1D instead of Linear.
    // We need to transpose their weights during export.
    private static readonly HashSet<string> Conv1dWeightNames = new()
    {
        "c_attn.weight",
        "c_proj.weight",
        "c_fc.weight"
    };

    /// <summary>
    /// Export the trained model to a HuggingFace-compatible directory.
    /// This directory can then be converted to GGUF using llama.cpp's converter.
    /// </summary>
    public void Export(Gpt2Model model, ModelConfig config, string outputDir,
                       string tokenizerDir)
    {
        Console.WriteLine($"\nExporting model to HuggingFace format: {outputDir}/");

        // Create the output directory
        Directory.CreateDirectory(outputDir);

        // =====================================================================
        // STEP 1: Collect and rename tensors
        // =====================================================================
        Console.WriteLine("\n  Collecting model parameters...");

        var tensors = new Dictionary<string, Tensor>();
        var lmHeadSaved = false;

        foreach (var (name, param) in model.named_parameters())
        {
            // Clone the parameter so we don't modify the model's weights
            var tensor = param.detach().clone();

            // Check if this weight needs to be transposed for Conv1D compatibility
            if (NeedsConv1dTranspose(name))
            {
                // Transpose: [out_features, in_features] → [in_features, out_features]
                Console.WriteLine($"    {name} → transposed for Conv1D compatibility");
                var transposed = tensor.t().contiguous();
                tensor.Dispose();
                tensor = transposed;
            }
            else
            {
                Console.WriteLine($"    {name}");
            }

            tensors[name] = tensor;

            if (name == "lm_head.weight")
                lmHeadSaved = true;
        }

        // If lm_head.weight wasn't in the named parameters (because of weight tying),
        // add it explicitly by copying from the token embeddings
        if (!lmHeadSaved)
        {
            Console.WriteLine("    lm_head.weight → copied from transformer.wte.weight (weight tying)");
            var wteWeight = tensors["transformer.wte.weight"];

            // For lm_head, we need Conv1D format too: [n_embd, vocab_size]
            // But lm_head in HuggingFace GPT-2 is NOT Conv1D, it's just the embedding weight.
            // So we DON'T transpose it.
            tensors["lm_head.weight"] = wteWeight.detach().clone();
        }

        Console.WriteLine($"\n  Total tensors: {tensors.Count}");

        // =====================================================================
        // STEP 2: Write model.safetensors
        // =====================================================================
        var safetensorsPath = Path.Combine(outputDir, "model.safetensors");
        SafeTensorsWriter.Write(safetensorsPath, tensors);

        // =====================================================================
        // STEP 3: Write config.json
        // =====================================================================
        // This must match the HuggingFace GPT-2 config format exactly.
        // The GGUF converter reads this to determine model architecture.
        var configJson = new Dictionary<string, object>
        {
            ["activation_function"] = "gelu_new",
            ["architectures"] = new[] { "GPT2LMHeadModel" },
            ["attn_pdrop"] = config.Dropout,
            ["bos_token_id"] = 50256,
            ["embd_pdrop"] = config.Dropout,
            ["eos_token_id"] = 50256,
            ["initializer_range"] = 0.02,
            ["layer_norm_epsilon"] = config.LayerNormEpsilon,
            ["model_type"] = "gpt2",
            ["n_ctx"] = config.BlockSize,
            ["n_embd"] = config.NEmbedding,
            ["n_head"] = config.NHead,
            ["n_inner"] = config.NInner,
            ["n_layer"] = config.NLayer,
            ["n_positions"] = config.BlockSize,
            ["resid_pdrop"] = config.Dropout,
            ["summary_activation"] = (object?)null!,
            ["summary_first_dropout"] = 0.1,
            ["summary_proj_to_labels"] = true,
            ["summary_type"] = "cls_index",
            ["summary_use_proj"] = true,
            ["task_specific_params"] = new Dictionary<string, object>
            {
                ["text-generation"] = new Dictionary<string, object>
                {
                    ["do_sample"] = true,
                    ["max_length"] = 50
                }
            },
            ["vocab_size"] = config.VocabSize
        };

        var configPath = Path.Combine(outputDir, "config.json");
        File.WriteAllText(configPath, JsonSerializer.Serialize(configJson, new JsonSerializerOptions
        {
            WriteIndented = true
        }));
        Console.WriteLine($"  Wrote {configPath}");

        // =====================================================================
        // STEP 4: Copy tokenizer files
        // =====================================================================
        CopyTokenizerFile(tokenizerDir, outputDir, "vocab.json");
        CopyTokenizerFile(tokenizerDir, outputDir, "merges.txt");

        // Write tokenizer_config.json
        var tokenizerConfig = new Dictionary<string, object>
        {
            ["model_max_length"] = config.BlockSize,
            ["tokenizer_class"] = "GPT2Tokenizer",
            ["bos_token"] = "<|endoftext|>",
            ["eos_token"] = "<|endoftext|>",
            ["unk_token"] = "<|endoftext|>"
        };
        var tokenizerConfigPath = Path.Combine(outputDir, "tokenizer_config.json");
        File.WriteAllText(tokenizerConfigPath, JsonSerializer.Serialize(tokenizerConfig, new JsonSerializerOptions
        {
            WriteIndented = true
        }));
        Console.WriteLine($"  Wrote {tokenizerConfigPath}");

        // Write special_tokens_map.json
        var specialTokens = new Dictionary<string, object>
        {
            ["bos_token"] = "<|endoftext|>",
            ["eos_token"] = "<|endoftext|>",
            ["unk_token"] = "<|endoftext|>"
        };
        var specialTokensPath = Path.Combine(outputDir, "special_tokens_map.json");
        File.WriteAllText(specialTokensPath, JsonSerializer.Serialize(specialTokens, new JsonSerializerOptions
        {
            WriteIndented = true
        }));
        Console.WriteLine($"  Wrote {specialTokensPath}");

        Console.WriteLine($"\n  Export complete! Directory: {outputDir}/");
    }

    /// <summary>
    /// Check if a tensor name corresponds to a weight that needs Conv1D transposition.
    /// In HuggingFace GPT-2, c_attn, c_proj (in attention), and c_fc use Conv1D
    /// which stores weights as [in_features, out_features] instead of
    /// Linear's [out_features, in_features].
    /// </summary>
    private static bool NeedsConv1dTranspose(string paramName)
    {
        // Check if the parameter name ends with any of the Conv1D weight patterns
        foreach (var conv1dName in Conv1dWeightNames)
        {
            if (paramName.EndsWith(conv1dName))
            {
                // Make sure it's a weight, not a bias (biases don't need transposition)
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Copy a tokenizer file from the tokenizer directory to the export directory.
    /// </summary>
    private static void CopyTokenizerFile(string tokenizerDir, string outputDir, string fileName)
    {
        var src = Path.Combine(tokenizerDir, fileName);
        var dst = Path.Combine(outputDir, fileName);

        if (File.Exists(src))
        {
            File.Copy(src, dst, overwrite: true);
            Console.WriteLine($"  Copied {fileName}");
        }
        else
        {
            Console.WriteLine($"  WARNING: {src} not found! GGUF conversion may fail.");
        }
    }
}
