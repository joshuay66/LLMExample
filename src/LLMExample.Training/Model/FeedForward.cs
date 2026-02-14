// =============================================================================
// FeedForward.cs - The Feed-Forward Network (FFN / MLP)
// =============================================================================
// Each transformer block has TWO main components:
//   1. Self-Attention (CausalSelfAttention) - tokens communicate with each other
//   2. Feed-Forward Network (this file) - each token is processed independently
//
// THE INTUITION:
// ==============
// After attention has gathered information from other tokens, the FFN processes
// that information at each position independently. Think of it as:
//   - Attention = "gathering ingredients from the fridge" (looking at context)
//   - FFN = "cooking with those ingredients" (computing features)
//
// ARCHITECTURE:
// =============
// The FFN is simply two linear layers with a GELU activation in between:
//   1. Expand: [n_embd] → [n_inner] (128 → 512, a 4x expansion)
//   2. Activate: GELU non-linearity (allows the network to learn non-linear patterns)
//   3. Contract: [n_inner] → [n_embd] (512 → 128, back to original size)
//
// The expansion-then-contraction pattern is called a "bottleneck" in reverse.
// The larger intermediate dimension gives the network more capacity to
// compute complex features before compressing back down.
//
// WHY GELU (NOT RELU)?
// ====================
// ReLU: f(x) = max(0, x) — hard cutoff at 0, can kill neurons permanently.
// GELU: f(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
//   — smooth approximation that allows small negative values through.
//   — every major LLM uses GELU because it trains more smoothly.
// =============================================================================

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using LLMExample.Training.Config;

namespace LLMExample.Training.Model;

public class FeedForward : Module<Tensor, Tensor>
{
    // Up-projection: expand from n_embd to n_inner (128 → 512)
    private readonly Linear c_fc;

    // Down-projection: compress from n_inner back to n_embd (512 → 128)
    private readonly Linear c_proj;

    // Dropout for regularization
    private readonly Dropout dropout;

    public FeedForward(ModelConfig config) : base(nameof(FeedForward))
    {
        c_fc = Linear(config.NEmbedding, config.NInner);
        c_proj = Linear(config.NInner, config.NEmbedding);
        dropout = Dropout(config.Dropout);

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass through the FFN.
    /// Input shape:  [batch_size, sequence_length, n_embd]
    /// Output shape: [batch_size, sequence_length, n_embd]
    /// </summary>
    public override Tensor forward(Tensor x)
    {
        // Step 1: Expand to higher dimension
        // [B, T, 128] → [B, T, 512]
        x = c_fc.forward(x);

        // Step 2: Apply GELU activation (the "new" GPT-2 variant with tanh approximation)
        // This introduces non-linearity — without it, stacking linear layers
        // would be equivalent to a single linear layer (no matter how many you stack).
        x = GeluNew(x);

        // Step 3: Project back down to model dimension
        // [B, T, 512] → [B, T, 128]
        x = c_proj.forward(x);

        // Step 4: Dropout
        x = dropout.forward(x);

        return x;
    }

    /// <summary>
    /// GPT-2's "new GELU" activation function.
    /// This is the tanh approximation of the Gaussian Error Linear Unit.
    ///
    /// Formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    ///
    /// It's almost identical to the exact GELU, but slightly faster to compute.
    /// The "0.044715" constant comes from a polynomial approximation of the
    /// Gaussian CDF (cumulative distribution function).
    /// </summary>
    private static Tensor GeluNew(Tensor x)
    {
        // sqrt(2/π) ≈ 0.7978845608
        const double sqrt2OverPi = 0.7978845608028654;
        const double coeff = 0.044715;

        // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        var inner = sqrt2OverPi * (x + coeff * x.pow(3));
        return 0.5 * x * (1.0 + inner.tanh());
    }
}
