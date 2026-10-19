# Attention Is All You Need: The Transformer Architecture

## 1. Executive Summary
The Transformer model, introduced by Vaswani et al. in 2017, completely revolutionized natural language processing (NLP) and artificial intelligence. Prior to the Transformer, recurrent neural networks (RNNs), Long Short-Term Memory networks (LSTMs), and gated recurrent units (GRUs) were the dominant approaches for sequence modeling and transduction problems such as machine translation. However, the sequential nature of RNNs precluded parallelization within training examples, creating severe bottlenecks when training on massive datasets.

The Transformer dispenses entirely with recurrence and convolutions, relying solely on an attention mechanism—specifically **Scaled Dot-Product Attention** and **Multi-Head Self-Attention**—to model global dependencies between input and output tokens regardless of distance.

## 2. Model Architecture
The Transformer adheres to a standard encoder-decoder topology:

### 2.1 The Encoder
The encoder is composed of a stack of $N = 6$ identical layers. Each layer contains two sub-layers:
1. **Multi-Head Self-Attention Mechanism**: Allows each position to attend to all positions in the previous layer.
2. **Position-wise Feed-Forward Network (FFN)**: Consists of two linear transformations with a ReLU or GELU activation in between:
   $$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
   The dimensionality of input and output is $d_{model} = 512$, and the inner-layer dimensionality is $d_{ff} = 2048$.

Residual connections surround each of the two sub-layers, followed by layer normalization:
$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

### 2.2 The Decoder
The decoder is also composed of a stack of $N = 6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer:
- **Masked Multi-Head Attention**: Modifies the self-attention sub-layer to prevent positions from attending to subsequent positions. This masking ensures that predictions for position $i$ depend only on known outputs at positions less than $i$.
- **Encoder-Decoder Cross-Attention**: Performs multi-head attention over the final representation output by the encoder stack.

## 3. Attention Mechanisms
Attention can be described as mapping a query and a set of key-value pairs to an output.

### 3.1 Scaled Dot-Product Attention
The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$. We compute the matrix of outputs as:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
The scaling factor $\frac{1}{\sqrt{d_k}}$ is essential: for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients.

### 3.2 Multi-Head Attention
Instead of performing a single attention function with $d_{model}$-dimensional queries, keys, and values, Multi-Head Attention projects queries, keys, and values $h$ times with different learned linear projections to $d_k$, $d_k$, and $d_v$ dimensions respectively ($h = 8$, $d_k = d_v = 64$).
Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions.

## 4. Positional Encoding
Because the model contains no recurrence and no convolution, positional encodings must be injected into input embeddings to provide token order information:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

## 5. Experimental Results and Impact
On the WMT 2014 English-to-German translation task, the Transformer achieved a state-of-the-art BLEU score of 28.4, outperforming the existing best models by more than 2.0 BLEU. Training required only 3.5 days on 8 NVIDIA P100 GPUs. The Transformer architecture forms the foundational bedrock for modern Large Language Models including GPT-4, BERT, PaLM, Llama, and Gemini.
