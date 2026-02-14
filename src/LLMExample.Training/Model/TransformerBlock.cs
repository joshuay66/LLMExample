// =============================================================================
// TransformerBlock.cs - One Layer of the Transformer
// =============================================================================
// A transformer is built by stacking multiple identical blocks. Each block
// has the same structure but learns different patterns at different levels
// of abstraction. Our model has 4 blocks stacked on top of each other.
//
// WHAT EACH BLOCK DOES:
// ====================
// 1. Self-Attention: Tokens look at each other and gather relevant information
// 2. Feed-Forward: Each token independently processes the gathered information
//
// TWO CRITICAL DESIGN PATTERNS:
// ============================
//
// 1. RESIDUAL CONNECTIONS (the "+" operations):
//    output = x + attention(norm(x))
//    Instead of: output = attention(x)
//
//    Why? As you stack many layers, gradients can shrink to zero during
//    backpropagation (the "vanishing gradient" problem). Residual connections
//    create a "highway" that lets gradients flow directly through the network.
//    The model can learn "what to add" to the existing representation rather
//    than computing the full representation from scratch at each layer.
//
//    This was the key insight from the 2015 ResNet paper that enabled
//    training very deep networks (100+ layers).
//
// 2. PRE-NORM (Layer Normalization BEFORE each sub-layer):
//    output = x + sublayer(LayerNorm(x))
//    Instead of the original: output = LayerNorm(x + sublayer(x))
//
//    GPT-2 uses "pre-norm" because it makes training more stable.
//    Layer normalization rescales the input to have mean=0 and variance=1,
//    which prevents the activations from growing or shrinking as they
//    pass through many layers.
// =============================================================================

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using LLMExample.Training.Config;

namespace LLMExample.Training.Model;

public class TransformerBlock : Module<Tensor, Tensor>
{
    // Layer normalization before the attention sub-layer
    private readonly LayerNorm ln_1;

    // The multi-head causal self-attention mechanism
    private readonly CausalSelfAttention attn;

    // Layer normalization before the feed-forward sub-layer
    private readonly LayerNorm ln_2;

    // The feed-forward network (MLP)
    private readonly FeedForward mlp;

    public TransformerBlock(ModelConfig config) : base(nameof(TransformerBlock))
    {
        ln_1 = LayerNorm(config.NEmbedding, eps: config.LayerNormEpsilon);
        attn = new CausalSelfAttention(config);
        ln_2 = LayerNorm(config.NEmbedding, eps: config.LayerNormEpsilon);
        mlp = new FeedForward(config);

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass through one transformer block.
    /// Input shape:  [batch_size, sequence_length, n_embd]
    /// Output shape: [batch_size, sequence_length, n_embd]
    ///
    /// The dimensionality never changes within a block — each block
    /// refines the representation while maintaining the same shape.
    /// </summary>
    public override Tensor forward(Tensor x)
    {
        // Pre-norm residual connection #1: Self-Attention
        //   1. Normalize the input (stabilizes training)
        //   2. Apply self-attention (tokens communicate)
        //   3. Add residual connection (preserve original information)
        x = x + attn.forward(ln_1.forward(x));

        // Pre-norm residual connection #2: Feed-Forward Network
        //   1. Normalize the input
        //   2. Apply feed-forward network (independent token processing)
        //   3. Add residual connection
        x = x + mlp.forward(ln_2.forward(x));

        return x;
    }
}
