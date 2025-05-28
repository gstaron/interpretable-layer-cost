# Interpretable Layer Cost Calculator

A web-based calculator for estimating the compute cost of training sparse autoencoders (SAEs) to make concepts inside Large Language Models (LLMs) interpretable. 

🔗 **Live Calculator**: https://gstaron.github.io/interpretable-layer-cost

## Overview

Understanding what neural networks "think" is one of the fundamental challenges in AI safety and interpretability. This project provides a practical tool for researchers to estimate the computational costs involved in training sparse autoencoders—a promising technique for extracting interpretable features from LLM activations.

## Background: Sparse Autoencoders for LLM Interpretability

### The Problem

Large Language Models process information through high-dimensional activation vectors that are difficult for humans to interpret. While these models demonstrate remarkable capabilities, understanding their internal representations remains a significant challenge for:

- AI safety research
- Model debugging and improvement
- Understanding failure modes
- Building more trustworthy AI systems

### The Solution: Sparse Autoencoders

Sparse autoencoders offer a promising approach to this interpretability challenge, as detailed in [Anthropic's research on monosemantic features](http://transformer-circuits.pub/2023/monosemantic-features/index.html#problem-setup). The key insight is that we can train a sparse autoencoder to:

1. **Encode** dense LLM activations into a sparse, higher-dimensional representation
2. **Decode** these sparse features back to approximate the original activations
3. **Interpret** individual features as meaningful concepts

### How It Works

The sparse autoencoder architecture consists of:

```
LLM Activation (dense) → Encoder → Sparse Features → Decoder → Reconstructed Activation
     [d_model]              [d_model × d_sae]    [d_sae]    [d_sae × d_model]      [d_model]
```

Where:
- `d_model`: Original LLM hidden dimension (typically 4096-8192)
- `d_sae`: Sparse autoencoder dimension (usually much larger, e.g., 8× to 64× the original)

## Using the Calculator

### Input Parameters

The calculator requires three key parameters:

1. **Hidden Dimension (d_model)**: The dimensionality of the LLM layer you want to interpret
   - Typical range: 4096-8192 for modern LLMs
   - Examples: GPT-4 layers, LLaMA hidden states, etc.

2. **Number of Training Samples**: The number of activation vectors needed to train the SAE
   - Rule of thumb: At least one sample per feature you want to discover
   - Typical range: 1M-100M+ samples
   - More samples → better feature discovery but higher cost

3. **Cost Per Weight Training (USD)**: The computational cost to train each parameter
   - Typical range: ~$0.0001 per weight on modern GPUs
   - Varies based on hardware, energy costs, and efficiency

### Calculation Method

The calculator estimates costs using this model:

```
Encoder Weights = d_model × d_sae
Decoder Weights = d_model × d_sae  
Total Weights = Encoder Weights + Decoder Weights
Training Cost = Total Weights × Cost Per Weight × Number of Samples
```

### Example Calculation

For a typical setup:
- Hidden Dimension: 4096
- Training Samples: 10,000,000
- Cost Per Weight: $0.0001

This yields:
- Encoder Weights: 4096 × 10M = 40.96B weights
- Decoder Weights: 4096 × 10M = 40.96B weights
- Total: ~82B weight updates
- Estimated Cost: ~$8,200 USD

## Research Context and Applications

### Current Research

This calculator is inspired by recent breakthroughs in mechanistic interpretability:

- [Anthropic's work on monosemantic features](http://transformer-circuits.pub/2023/monosemantic-features/index.html)
- Research on superposition and feature extraction in neural networks
- Advances in sparse coding for neural network interpretability

### Open Source Opportunities

The interpretability field offers exciting opportunities for open-source contributions:

1. **Training SAEs on Public Models**: Apply these techniques to openly available models like LLaMA, Mistral, or other public weights
2. **Scaling Law Research**: Investigate relationships between training data, model size, and feature quality
3. **Feature Analysis**: Develop tools for analyzing and visualizing discovered features
4. **Cost Optimization**: Research more efficient training methods to reduce computational requirements

### Future Improvements

Planned enhancements to this calculator:

- [ ] Integration with scaling laws for more accurate cost prediction
- [ ] Support for different SAE architectures (TopK, Gated, etc.)
- [ ] Hardware-specific cost models (A100, H100, etc.)
- [ ] Feature quality vs. cost trade-off analysis
- [ ] Integration with existing interpretability frameworks

## Installation and Usage

### Quick Start

1. Open the calculator at: https://gstaron.github.io/interpretable-layer-cost
2. Enter your parameters
3. Click "Calculate Cost" to get your estimate

### Local Development

```bash
# Clone the repository
git clone https://github.com/gstaron/interpretable-layer-cost.git
cd interpretable-layer-cost

# Open in your browser
open index.html
```

No build process required—it's a simple HTML/JavaScript calculator!

## Contributing

We welcome contributions! Areas where help is needed:

- More sophisticated cost models
- Support for different SAE variants
- Integration with existing interpretability tools
- Documentation improvements
- User interface enhancements

## Related Work and References

- [Anthropic: Towards Monosemantic Features](http://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Mechanistic Interpretability Research](https://transformer-circuits.pub/)
- [OpenAI: Sparse Autoencoders for Language Models](https://openai.com/research)

## License

This project is released into the public domain under the Unlicense. See [LICENSE](LICENSE) for details.

---

**Disclaimer**: Cost estimates are approximations based on simplified models. Actual training costs may vary significantly based on implementation details, hardware efficiency, optimization techniques, and other factors. Always validate estimates with your specific setup before committing to large-scale training runs.
