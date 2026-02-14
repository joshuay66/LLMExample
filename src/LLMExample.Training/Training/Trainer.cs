// =============================================================================
// Trainer.cs - The Training Loop
// =============================================================================
// This is where the model actually LEARNS. The training loop repeats these
// steps thousands of times:
//
//   1. SAMPLE: Grab a random batch of text from the dataset
//   2. FORWARD: Run the text through the model to get predictions
//   3. LOSS: Measure how wrong the predictions are (cross-entropy loss)
//   4. BACKWARD: Compute gradients — how much each weight contributed to error
//   5. UPDATE: Adjust weights in the direction that reduces the error
//   6. REPEAT: Go back to step 1
//
// THE KEY CONCEPTS:
// =================
//
// LOSS FUNCTION (Cross-Entropy):
//   The model outputs a probability distribution over all 50,257 tokens
//   at each position. Cross-entropy measures how different the model's
//   prediction is from the actual next token. Lower loss = better predictions.
//   - Random guessing: loss ≈ ln(50257) ≈ 10.82
//   - Perfect prediction: loss = 0
//   - Decent model: loss ≈ 3-5 (for this small model on limited data)
//
// GRADIENT DESCENT:
//   The "backward" pass computes partial derivatives (gradients) of the loss
//   with respect to every weight in the model. These gradients tell us:
//   "If I increase this weight by a tiny amount, how much does the loss change?"
//   We then move each weight in the direction that DECREASES the loss.
//
// ADAMW OPTIMIZER:
//   Plain gradient descent is slow and unstable. AdamW improves on it by:
//   - Maintaining a running average of gradients (momentum — smooths out noise)
//   - Maintaining a running average of squared gradients (adapts step size per-weight)
//   - Weight decay: gently shrinks weights toward zero (regularization)
// =============================================================================

using TorchSharp;
using static TorchSharp.torch;
using LLMExample.Training.Config;
using LLMExample.Training.Data;
using LLMExample.Training.Model;

namespace LLMExample.Training.Training;

public class Trainer
{
    private readonly Gpt2Model _model;
    private readonly TextDataset _dataset;
    private readonly ModelConfig _config;
    private readonly Device _device;

    public Trainer(Gpt2Model model, TextDataset dataset, ModelConfig config, Device device)
    {
        _model = model;
        _dataset = dataset;
        _config = config;
        _device = device;
    }

    /// <summary>
    /// Run the complete training loop.
    /// </summary>
    public void Train()
    {
        // Put the model in training mode. This enables dropout
        // (which is disabled during inference/generation).
        _model.train();

        // Initialize the AdamW optimizer.
        // AdamW = Adam with decoupled Weight decay.
        //
        // Key hyperparameters:
        //   lr (learning rate): Step size for weight updates. 3e-4 is a good default.
        //   weight_decay: L2 regularization strength. Prevents overfitting.
        //   beta1, beta2: Momentum parameters (defaults: 0.9, 0.999)
        //     beta1 controls how much past gradients influence the current update
        //     beta2 controls the adaptive learning rate per parameter
        var optimizer = optim.AdamW(
            _model.parameters(),
            lr: _config.LearningRate,
            weight_decay: _config.WeightDecay
        );

        Console.WriteLine($"\nStarting training for {_config.MaxSteps} steps...");
        Console.WriteLine($"  Batch size: {_config.BatchSize}");
        Console.WriteLine($"  Block size: {_config.BlockSize}");
        Console.WriteLine($"  Learning rate: {_config.LearningRate}");
        Console.WriteLine($"  Tokens per step: {_config.BatchSize * _config.BlockSize:N0}");
        Console.WriteLine();

        var startTime = DateTime.UtcNow;

        for (int step = 1; step <= _config.MaxSteps; step++)
        {
            // =================================================================
            // STEP 1: Get a random batch of training data
            // =================================================================
            // Returns (input, target) tensors of shape [BatchSize, BlockSize]
            // target[i][j] = input[i][j+1] (the next token)
            var (input, target) = _dataset.GetBatch(_config.BatchSize, _device);

            // =================================================================
            // STEP 2: Forward pass — run input through the model
            // =================================================================
            // The model processes the input tokens and outputs logits
            // (unnormalized scores) for every possible next token at each position.
            // logits shape: [BatchSize, BlockSize, VocabSize]
            var logits = _model.forward(input);

            // =================================================================
            // STEP 3: Compute the loss (cross-entropy)
            // =================================================================
            // Cross-entropy loss compares the model's predicted probability
            // distribution against the actual next token.
            //
            // We need to reshape:
            //   logits: [B, T, V] → [B*T, V] (flatten batch and sequence)
            //   target: [B, T]    → [B*T]     (flatten to 1D)
            //
            // This is because PyTorch's cross_entropy expects:
            //   predictions: [num_examples, num_classes]
            //   targets: [num_examples]
            var loss = nn.functional.cross_entropy(
                logits.view(-1, _config.VocabSize), // [B*T, V]
                target.view(-1)                      // [B*T]
            );

            // =================================================================
            // STEP 4: Backward pass — compute gradients
            // =================================================================
            // First, zero out gradients from the previous step.
            // Without this, gradients would accumulate across steps!
            optimizer.zero_grad();

            // Compute gradients using backpropagation.
            // This is the chain rule of calculus applied automatically:
            // for each weight w, compute ∂loss/∂w
            loss.backward();

            // =================================================================
            // STEP 5: Gradient clipping
            // =================================================================
            // If gradients are too large ("exploding gradients"), scale them down.
            // This prevents a single bad batch from causing wild weight updates
            // that destabilize training. Max norm of 1.0 is standard.
            nn.utils.clip_grad_norm_(_model.parameters(), _config.MaxGradNorm);

            // =================================================================
            // STEP 6: Update weights
            // =================================================================
            // The optimizer adjusts every weight in the model based on the
            // computed gradients, using the AdamW algorithm.
            optimizer.step();

            // =================================================================
            // STEP 7: Logging
            // =================================================================
            if (step % _config.LogEveryNSteps == 0 || step == 1)
            {
                var elapsed = DateTime.UtcNow - startTime;
                var tokensPerSec = (long)step * _config.BatchSize * _config.BlockSize / elapsed.TotalSeconds;
                var lossValue = loss.item<float>();

                Console.WriteLine(
                    $"  Step {step,5}/{_config.MaxSteps} | " +
                    $"Loss: {lossValue:F4} | " +
                    $"Tokens/sec: {tokensPerSec:N0} | " +
                    $"Elapsed: {elapsed:hh\\:mm\\:ss}");
            }

            // =================================================================
            // STEP 8: Dispose tensors to prevent memory leaks
            // =================================================================
            // CRITICAL in TorchSharp! Unlike Python's PyTorch which uses
            // reference counting for immediate cleanup, C#'s garbage collector
            // is non-deterministic. Without explicit disposal, GPU/CPU memory
            // would grow unbounded and eventually crash.
            input.Dispose();
            target.Dispose();
            logits.Dispose();
            loss.Dispose();
        }

        var totalTime = DateTime.UtcNow - startTime;
        Console.WriteLine($"\nTraining complete! Total time: {totalTime:hh\\:mm\\:ss}");
    }
}
