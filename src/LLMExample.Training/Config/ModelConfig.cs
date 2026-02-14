// =============================================================================
// ModelConfig.cs - Model & Training Hyperparameters
// =============================================================================
// This file defines ALL the knobs you can turn when building and training
// your language model. Think of it as the "blueprint specifications" --
// it determines the model's capacity (how much it can learn) and the
// training behavior (how it learns).
//
// KEY CONCEPT: There's always a tradeoff between model size and training
// speed. Larger models can learn more complex patterns but take longer
// to train and need more memory. Our settings here create a tiny model
// (~7.3 million parameters) that trains in under an hour on a Mac.
// For comparison, GPT-2 "small" has 124 million parameters,
// and GPT-4 is rumored to have over a trillion.
// =============================================================================

namespace LLMExample.Training.Config;

public class ModelConfig
{
    // =========================================================================
    // MODEL ARCHITECTURE
    // These parameters define the structure of the neural network.
    // Changing any of these creates a fundamentally different model.
    // =========================================================================

    /// <summary>
    /// The number of unique tokens (words/subwords) the model knows.
    /// We use the standard GPT-2 BPE vocabulary with 50,257 tokens.
    /// This is the largest contributor to our parameter count because
    /// each token gets its own embedding vector of size NEmbedding.
    /// Parameters from this: VocabSize × NEmbedding = 50,257 × 128 = ~6.4M
    /// </summary>
    public int VocabSize { get; init; } = 50257;

    /// <summary>
    /// The size of each token's "meaning" vector (the embedding dimension).
    /// Every token and position gets represented as a vector of this size.
    /// Larger = the model can represent more nuanced meanings, but uses more memory.
    /// GPT-2 small uses 768; we use 128 to keep things fast.
    /// </summary>
    public int NEmbedding { get; init; } = 128;

    /// <summary>
    /// Number of attention heads. Multi-head attention lets the model
    /// attend to different aspects of the input simultaneously.
    /// For example, one head might learn syntax, another semantics.
    /// Each head operates on NEmbedding/NHead = 128/4 = 32 dimensions.
    /// GPT-2 small uses 12 heads; we use 4.
    /// </summary>
    public int NHead { get; init; } = 4;

    /// <summary>
    /// Number of transformer blocks (layers) stacked on top of each other.
    /// More layers = deeper understanding but slower training.
    /// Each layer refines the representation from the previous layer.
    /// GPT-2 small uses 12 layers; we use 4.
    /// </summary>
    public int NLayer { get; init; } = 4;

    /// <summary>
    /// The maximum sequence length (context window) the model can process.
    /// The model can "see" this many tokens at once when making predictions.
    /// Longer context = can understand longer passages, but uses quadratically
    /// more memory in the attention mechanism (256² = 65,536 attention scores).
    /// GPT-2 uses 1024; we use 256 to keep memory manageable.
    /// </summary>
    public int BlockSize { get; init; } = 256;

    /// <summary>
    /// Hidden dimension of the feed-forward network inside each transformer block.
    /// Convention is 4× the embedding dimension. The FFN first expands to this
    /// size (learning complex features) then projects back down to NEmbedding.
    /// </summary>
    public int NInner { get; init; } = 512;

    /// <summary>
    /// Dropout rate: the probability of randomly "turning off" neurons during
    /// training. This prevents overfitting by forcing the model to not rely
    /// on any single neuron too much. Set to 0 during inference (generation).
    /// Typical range: 0.0 to 0.3. Higher = more regularization.
    /// </summary>
    public float Dropout { get; init; } = 0.1f;

    /// <summary>
    /// A tiny constant added to layer normalization to prevent division by zero.
    /// You'll almost never need to change this. Standard value across all GPT models.
    /// </summary>
    public float LayerNormEpsilon { get; init; } = 1e-5f;

    // =========================================================================
    // TRAINING HYPERPARAMETERS
    // These control HOW the model learns, not WHAT it learns.
    // =========================================================================

    /// <summary>
    /// Learning rate: how big of a step the optimizer takes when updating weights.
    /// Too high = training is unstable (loss spikes or diverges).
    /// Too low = training is painfully slow.
    /// 3e-4 is a solid default for Adam-family optimizers on small models.
    /// </summary>
    public float LearningRate { get; init; } = 3e-4f;

    /// <summary>
    /// How many training examples to process at once before updating weights.
    /// Larger batch = more stable gradients but uses more memory.
    /// Smaller batch = noisier gradients but can sometimes generalize better.
    /// 16 is a good starting point for our model size.
    /// </summary>
    public int BatchSize { get; init; } = 16;

    /// <summary>
    /// Maximum number of training steps (weight updates).
    /// One step = process one batch → compute loss → update weights.
    /// 5000 steps × batch size 16 × block size 256 = ~20M tokens processed.
    /// </summary>
    public int MaxSteps { get; init; } = 5000;

    /// <summary>
    /// How often to print training progress (every N steps).
    /// </summary>
    public int LogEveryNSteps { get; init; } = 100;

    /// <summary>
    /// Weight decay for the AdamW optimizer. This is a form of regularization
    /// that gently pushes weights toward zero, preventing them from growing
    /// too large. Helps the model generalize to new text.
    /// </summary>
    public float WeightDecay { get; init; } = 0.01f;

    /// <summary>
    /// Maximum gradient norm for gradient clipping. During training, if the
    /// gradients become too large (which can cause "exploding gradients"),
    /// they are scaled down to this maximum magnitude.
    /// </summary>
    public float MaxGradNorm { get; init; } = 1.0f;

    // =========================================================================
    // FILE PATHS
    // =========================================================================

    /// <summary>
    /// Directory containing your .txt training files.
    /// All .txt files in this directory will be loaded and concatenated.
    /// </summary>
    public string DataDirectory { get; init; } = "data/training";

    /// <summary>
    /// Directory where the HuggingFace-format model will be exported.
    /// This directory is then fed to the Python GGUF conversion script.
    /// </summary>
    public string ExportDirectory { get; init; } = "exported_model";

    /// <summary>
    /// Directory containing the GPT-2 tokenizer files (vocab.json, merges.txt).
    /// These files define how text is split into tokens.
    /// </summary>
    public string TokenizerDirectory { get; init; } = "tokenizer";
}
