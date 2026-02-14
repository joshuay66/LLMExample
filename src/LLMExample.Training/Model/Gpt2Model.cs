// =============================================================================
// Gpt2Model.cs - The Complete GPT-2 Language Model
// =============================================================================
// This is the top-level model that ties everything together. Here's the
// complete flow of data through the model:
//
//   Input: Token IDs [batch, seq_len] (e.g., [16, 256])
//          ↓
//   Token Embedding: Look up each token's learned vector representation
//          ↓ [batch, seq_len, n_embd]
//   Position Embedding: Add positional information (where each token is)
//          ↓ [batch, seq_len, n_embd]
//   Dropout: Randomly zero out some values (regularization)
//          ↓ [batch, seq_len, n_embd]
//   Transformer Block 1: Attention + FFN
//          ↓ [batch, seq_len, n_embd]
//   Transformer Block 2: Attention + FFN
//          ↓ [batch, seq_len, n_embd]
//   Transformer Block 3: Attention + FFN
//          ↓ [batch, seq_len, n_embd]
//   Transformer Block 4: Attention + FFN
//          ↓ [batch, seq_len, n_embd]
//   Final Layer Norm: Normalize the output
//          ↓ [batch, seq_len, n_embd]
//   Language Model Head: Project back to vocabulary size
//          ↓ [batch, seq_len, vocab_size]
//   Output: Logits (unnormalized scores) for every token in the vocabulary
//
// The logits at position i represent the model's prediction for what token
// comes at position i+1. During training, we compare these predictions
// against the actual next tokens using cross-entropy loss.
//
// TENSOR NAMING FOR EXPORT:
// =========================
// The nested module structure here (Gpt2Model → transformer → components)
// is carefully designed so that TorchSharp automatically generates tensor
// names matching the HuggingFace GPT-2 convention:
//   transformer.wte.weight, transformer.h.0.attn.c_attn.weight, etc.
// This is CRITICAL for GGUF conversion to work correctly.
// =============================================================================

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using LLMExample.Training.Config;

namespace LLMExample.Training.Model;

/// <summary>
/// The inner transformer module. This is a separate class so that TorchSharp
/// prefixes all its parameter names with "transformer." — matching HuggingFace's
/// naming convention for GPT-2.
/// </summary>
public class Gpt2Transformer : Module<Tensor, Tensor>
{
    // Token embeddings: maps each token ID to a learned vector of size n_embd.
    // This is a lookup table with 50,257 rows (one per token) and 128 columns.
    // "cat" might map to [0.23, -0.15, 0.87, ...] while "dog" maps to
    // something nearby in this vector space (since they're semantically similar).
    private readonly Embedding wte;

    // Position embeddings: maps each position (0-255) to a learned vector.
    // This is how the model knows WHERE a token appears in the sequence.
    // Without this, the model would see "the cat sat" the same as "sat cat the".
    // Unlike sinusoidal position encodings (used in the original Transformer paper),
    // GPT-2 LEARNS the position embeddings during training.
    private readonly Embedding wpe;

    // The stack of transformer blocks. Each block refines the representation.
    // Earlier blocks tend to learn simpler patterns (grammar, syntax).
    // Later blocks tend to learn more complex patterns (semantics, reasoning).
    private readonly ModuleList<TransformerBlock> h;

    // Final layer normalization applied after all transformer blocks.
    private readonly LayerNorm ln_f;

    // Dropout applied to the sum of token + position embeddings.
    private readonly Dropout drop;

    private readonly int _blockSize;

    public Gpt2Transformer(ModelConfig config) : base("transformer")
    {
        _blockSize = config.BlockSize;

        wte = Embedding(config.VocabSize, config.NEmbedding);
        wpe = Embedding(config.BlockSize, config.NEmbedding);

        // Create the stack of transformer blocks
        var blocks = new List<TransformerBlock>();
        for (int i = 0; i < config.NLayer; i++)
        {
            blocks.Add(new TransformerBlock(config));
        }
        h = new ModuleList<TransformerBlock>(blocks.ToArray());

        ln_f = LayerNorm(config.NEmbedding, eps: config.LayerNormEpsilon);
        drop = Dropout(config.Dropout);

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass through the transformer.
    /// Input:  token IDs [batch, seq_len]
    /// Output: hidden states [batch, seq_len, n_embd]
    /// </summary>
    public override Tensor forward(Tensor idx)
    {
        var device = idx.device;
        long T = idx.shape[1]; // sequence length

        // Create position indices: [0, 1, 2, ..., T-1]
        var pos = torch.arange(0, T, dtype: ScalarType.Int64, device: device)
            .unsqueeze(0); // [1, T] — will broadcast across the batch

        // Look up embeddings for tokens and positions
        var tokEmb = wte.forward(idx); // [B, T, n_embd]
        var posEmb = wpe.forward(pos); // [1, T, n_embd] — broadcasts to [B, T, n_embd]

        // Combine: the representation of a token is its meaning (tokEmb) + its
        // position (posEmb). This is simply element-wise addition.
        var x = drop.forward(tokEmb + posEmb);

        // Pass through each transformer block
        foreach (var block in h)
        {
            x = block.forward(x);
        }

        // Final layer normalization
        x = ln_f.forward(x);

        // Clean up
        pos.Dispose();
        tokEmb.Dispose();
        posEmb.Dispose();

        return x; // [B, T, n_embd]
    }
}

/// <summary>
/// The complete GPT-2 Language Model with a prediction head.
/// This wraps the transformer and adds the final projection to vocabulary logits.
/// </summary>
public class Gpt2Model : Module<Tensor, Tensor>
{
    private readonly Gpt2Transformer transformer;

    // The language model "head": projects from n_embd back to vocab_size.
    // This produces a score for every possible next token.
    // Its weight is TIED to the token embedding (wte) — see InitWeightTying().
    // Weight tying is a common technique that:
    //   1. Saves parameters (6.4M in our case)
    //   2. Enforces consistency: tokens with similar embeddings should have
    //      similar prediction scores
    private readonly Linear lm_head;

    private readonly ModelConfig _config;

    public Gpt2Model(ModelConfig config) : base(nameof(Gpt2Model))
    {
        _config = config;
        transformer = new Gpt2Transformer(config);

        // lm_head projects [n_embd] → [vocab_size] to produce next-token predictions
        // bias: false because GPT-2 doesn't use bias in the output head
        lm_head = Linear(config.NEmbedding, config.VocabSize, hasBias: false);

        RegisterComponents();

        // Apply weight initialization (Xavier/Glorot uniform, similar to GPT-2)
        InitWeights();

        // Tie the output projection weights to the token embeddings
        InitWeightTying();
    }

    /// <summary>
    /// Forward pass: token IDs → next-token logits.
    /// Input:  [batch, seq_len] token IDs
    /// Output: [batch, seq_len, vocab_size] logits (unnormalized prediction scores)
    /// </summary>
    public override Tensor forward(Tensor idx)
    {
        // Get hidden states from the transformer
        var hidden = transformer.forward(idx); // [B, T, n_embd]

        // Project to vocabulary size to get prediction scores
        var logits = lm_head.forward(hidden); // [B, T, vocab_size]

        hidden.Dispose();
        return logits;
    }

    /// <summary>
    /// Initialize weights using a scheme similar to the original GPT-2.
    /// Good initialization is crucial — if weights start too large or too small,
    /// training may fail to converge.
    /// </summary>
    private void InitWeights()
    {
        foreach (var (name, param) in named_parameters())
        {
            if (name.EndsWith(".weight") && param.dim() >= 2)
            {
                // Xavier uniform initialization for weight matrices
                // This sets the initial scale based on the layer dimensions,
                // keeping the variance of activations roughly constant across layers.
                nn.init.xavier_uniform_(param);
            }
            else if (name.EndsWith(".bias"))
            {
                // Initialize biases to zero
                nn.init.zeros_(param);
            }
        }
    }

    /// <summary>
    /// Tie the lm_head weight to the token embedding weight.
    /// Both matrices are [vocab_size, n_embd], so we make them share
    /// the same underlying data. When one is updated during training,
    /// the other automatically reflects the change.
    /// </summary>
    private void InitWeightTying()
    {
        // Access the wte embedding weight from the transformer module
        // and set it as the lm_head's weight.
        // This means lm_head.weight and transformer.wte.weight point to
        // the same tensor in memory.
        lm_head.weight = transformer.get_parameter("wte.weight")!;
    }

    /// <summary>
    /// Get the model configuration (needed by the exporter).
    /// </summary>
    public ModelConfig Config => _config;
}
