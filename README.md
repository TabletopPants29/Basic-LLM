# Character-Level Transformer Language Model

A lightweight, from-scratch implementation of a Transformer-based language model using PyTorch. This project trains a generative model at the character level, learning to predict the next character in a sequence based on the context of previous characters. It features a complete Decoder-only Transformer architecture, heavily inspired by modern Large Language Models (LLMs).

## Features

*   **Built in PyTorch:** Utilizes core PyTorch modules (`torch.nn`) for tensor operations and neural network layers.
*   **Multi-Head Self-Attention:** Implements parallel attention heads to capture complex character relationships and dependencies.
*   **Custom Tokenizer:** Automatically builds a character-level vocabulary mapping (`stoi` and `itos`) from the provided dataset.
*   **GPU Acceleration:** Automatically detects and utilizes CUDA for faster training if available.
*   **Text Generation:** Includes a sampling function to generate novel text sequences after training.

---

## Architecture & Hyperparameters

The model uses a standard Transformer Decoder architecture with positional embeddings, multi-head attention blocks, feed-forward layers, and layer normalization. 

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Batch Size** | `64` | Number of independent sequences processed in parallel |
| **Block Size** | `256` | Maximum context length for predictions |
| **Max Iterations** | `5000` | Total number of training steps |
| **Learning Rate** | `3e-4` | Step size for the AdamW optimizer |
| **Embedding Dims** | `384` | Size of the token and positional embeddings |
| **Attention Heads** | `6` | Number of parallel self-attention heads |
| **Layers** | `6` | Number of Transformer blocks |
| **Dropout** | `0.2` | Dropout probability to prevent overfitting |

---

## Getting Started

### Prerequisites
Ensure you have Python installed along with PyTorch. You can install PyTorch via [pytorch.org](https://pytorch.org/get-started/locally/).

### Installation
1. Clone this repository to your local machine.
2. Place your training data inside the root directory and name it `text.txt`. (A popular choice for character-level models is the "Tiny Shakespeare" dataset).

### Usage
To train the model and generate text, simply run the Python script from your terminal:

```bash
python train.py
