// =============================================================================
// CausalSelfAttention.cs - The Heart of the Transformer
// =============================================================================
// Self-attention is the mechanism that allows each token to "look at" other
// tokens in the sequence and decide how much to pay attention to each one.
//
// THE INTUITION:
// ==============
// Imagine reading the sentence: "The cat sat on the mat because it was tired."
// When processing the word "it", the model needs to figure out that "it"
// refers to "the cat" (not "the mat"). Self-attention does this by computing
// a relevance score between every pair of tokens.
//
// THE THREE VECTORS: Q, K, V (Query, Key, Value)
// ===============================================
// For each token, we compute three vectors:
//   - Query (Q): "What am I looking for?"
//   - Key (K):   "What do I contain?"
//   - Value (V): "What information do I provide if selected?"
//
// The attention score between token_i and token_j is:
//   score = Q_i · K_j / sqrt(head_dim)
//
// High score = token_i should pay attention to token_j.
// These scores become weights (via softmax) that blend the Value vectors.
//
// WHY "CAUSAL"?
// =============
// In a language model, when predicting the next token, we can only look at
// PAST tokens, not future ones. "Causal" means we mask out future positions
// so the model can't cheat by looking ahead. This is done with a triangular
// mask that sets future positions to negative infinity before softmax.
//
// MULTI-HEAD ATTENTION:
// ====================
// Instead of one set of Q/K/V, we use multiple "heads" (4 in our case).
// Each head can learn to attend to different things:
//   Head 1 might learn: syntax (subject-verb agreement)
//   Head 2 might learn: semantics (word meaning relationships)
//   Head 3 might learn: positional patterns (nearby words)
//   Head 4 might learn: something else entirely
// The heads' outputs are concatenated and projected back to the model dimension.
// =============================================================================

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using LLMExample.Training.Config;

namespace LLMExample.Training.Model;

public class CausalSelfAttention : Module<Tensor, Tensor>
{
    // Combined Q, K, V projection. Instead of three separate matrix multiplications,
    // GPT-2 does one big matmul that produces all three at once (more efficient).
    // Input: [batch, seq_len, n_embd] → Output: [batch, seq_len, 3 * n_embd]
    private readonly Linear c_attn;

    // Output projection. After attention, project back to the model dimension.
    // Input: [batch, seq_len, n_embd] → Output: [batch, seq_len, n_embd]
    private readonly Linear c_proj;

    // Dropout layers for regularization
    private readonly Dropout attn_drop;
    private readonly Dropout resid_drop;

    private readonly int _nHead;
    private readonly int _nEmbd;
    private readonly int _headDim;

    public CausalSelfAttention(ModelConfig config) : base(nameof(CausalSelfAttention))
    {
        _nHead = config.NHead;
        _nEmbd = config.NEmbedding;

        // Each head gets an equal slice of the embedding dimension
        // 128 / 4 = 32 dimensions per head
        _headDim = config.NEmbedding / config.NHead;

        // Combined QKV projection: one matrix multiply produces Q, K, and V
        // This is more efficient than three separate projections because
        // modern GPUs are better at one large matrix multiply than three small ones.
        c_attn = Linear(config.NEmbedding, 3 * config.NEmbedding);

        // Output projection after concatenating all heads
        c_proj = Linear(config.NEmbedding, config.NEmbedding);

        attn_drop = Dropout(config.Dropout);
        resid_drop = Dropout(config.Dropout);

        // Register all sub-modules so TorchSharp tracks their parameters.
        // This is essential for the optimizer to find and update all weights.
        RegisterComponents();
    }

    /// <summary>
    /// Forward pass through the attention mechanism.
    /// Input shape:  [batch_size, sequence_length, n_embd]
    /// Output shape: [batch_size, sequence_length, n_embd]
    /// </summary>
    public override Tensor forward(Tensor x)
    {
        // x shape: [B, T, C] where B=batch, T=sequence length, C=embedding dim
        var shape = x.shape;
        long B = shape[0]; // batch size
        long T = shape[1]; // sequence length
        long C = shape[2]; // embedding dimension (n_embd)

        // =====================================================================
        // STEP 1: Compute Q, K, V with a single matrix multiplication
        // =====================================================================
        // c_attn projects [B, T, n_embd] → [B, T, 3*n_embd]
        // We then split into three equal chunks: Q, K, V each of shape [B, T, n_embd]
        var qkv = c_attn.forward(x);
        var chunks = qkv.chunk(3, dim: -1); // Split along the last dimension
        var q = chunks[0]; // Query:  [B, T, n_embd]
        var k = chunks[1]; // Key:    [B, T, n_embd]
        var v = chunks[2]; // Value:  [B, T, n_embd]

        // =====================================================================
        // STEP 2: Reshape for multi-head attention
        // =====================================================================
        // Reshape from [B, T, n_embd] to [B, T, nHead, headDim]
        // Then transpose to [B, nHead, T, headDim] so each head is a separate "batch"
        //
        // Think of it as splitting the 128-dim vector into 4 groups of 32,
        // where each group is processed by its own attention head.
        q = q.view(B, T, _nHead, _headDim).transpose(1, 2); // [B, nHead, T, headDim]
        k = k.view(B, T, _nHead, _headDim).transpose(1, 2);
        v = v.view(B, T, _nHead, _headDim).transpose(1, 2);

        // =====================================================================
        // STEP 3: Compute attention scores
        // =====================================================================
        // Attention(Q, K, V) = softmax(Q·Kᵀ / √headDim) · V
        //
        // Q·Kᵀ gives us a [T, T] matrix where entry (i, j) is how much
        // token i should attend to token j.
        //
        // We divide by √headDim to prevent the dot products from growing
        // too large (which would push softmax to extremes, making gradients tiny).
        var scale = 1.0 / Math.Sqrt(_headDim);
        var att = torch.matmul(q, k.transpose(-2, -1)) * scale;
        // att shape: [B, nHead, T, T]

        // =====================================================================
        // STEP 4: Apply the causal mask
        // =====================================================================
        // Create a lower-triangular mask: position i can only attend to positions ≤ i.
        // Future positions get -infinity, which becomes 0 after softmax.
        //
        //   [1, -inf, -inf, -inf]     [1.0, 0.0, 0.0, 0.0]
        //   [1,    1, -inf, -inf]  →  [0.5, 0.5, 0.0, 0.0]  (after softmax)
        //   [1,    1,    1, -inf]     [0.3, 0.3, 0.3, 0.0]
        //   [1,    1,    1,    1]     [0.25,0.25,0.25,0.25]
        //
        var mask = torch.ones(T, T, device: x.device)
            .tril()  // Lower triangular matrix (1s on and below the diagonal)
            .view(1, 1, T, T); // Expand for broadcasting: [1, 1, T, T]

        // Where mask is 0 (future positions), set attention score to -infinity
        att = att.masked_fill(mask.eq(0), float.NegativeInfinity);

        // =====================================================================
        // STEP 5: Softmax → Attention weights
        // =====================================================================
        // Softmax normalizes each row to sum to 1.0, creating a probability
        // distribution over which tokens to attend to.
        att = torch.nn.functional.softmax(att, dim: -1);
        att = attn_drop.forward(att); // Dropout on attention weights

        // =====================================================================
        // STEP 6: Weighted sum of values
        // =====================================================================
        // Multiply attention weights by values to get a weighted combination
        // of information from attended tokens.
        // [B, nHead, T, T] × [B, nHead, T, headDim] → [B, nHead, T, headDim]
        var y = torch.matmul(att, v);

        // =====================================================================
        // STEP 7: Concatenate heads and project
        // =====================================================================
        // Transpose back: [B, nHead, T, headDim] → [B, T, nHead, headDim]
        // Then reshape to concatenate heads: [B, T, nHead * headDim] = [B, T, n_embd]
        y = y.transpose(1, 2).contiguous().view(B, T, C);

        // Final linear projection and dropout
        y = c_proj.forward(y);
        y = resid_drop.forward(y);

        // Clean up intermediate tensors to prevent memory buildup.
        // TorchSharp (unlike Python PyTorch) relies on explicit disposal
        // because C#'s garbage collector is non-deterministic.
        qkv.Dispose();
        foreach (var chunk in chunks) chunk.Dispose();
        q.Dispose();
        k.Dispose();
        v.Dispose();
        att.Dispose();
        mask.Dispose();

        return y;
    }
}
