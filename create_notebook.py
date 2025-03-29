import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Character-Level Tokenization and Embeddings\n\nThis notebook demonstrates character-level tokenization and embeddings using the Shakespeare dataset, following Andrej Karpathy's approach."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": ["import torch\nimport numpy as np\nfrom pathlib import Path"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Load and Inspect the Data"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Read the Shakespeare text\n",
                "with open('../data/shakespeare.txt', 'r', encoding='utf-8') as f:\n",
                "    text = f.read()\n\n",
                "print(f'Length of dataset in characters: {len(text)}')\n",
                "print('\\nFirst 1000 characters:\\n')\n",
                "print(text[:1000])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Character-Level Tokenization\n\nIn character-level tokenization, each unique character in the text becomes a token. This is the simplest form of tokenization."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Get all unique characters in the text\n",
                "chars = sorted(list(set(text)))\n",
                "vocab_size = len(chars)\n\n",
                "print(f'Vocabulary size (unique characters): {vocab_size}')\n",
                "print('\\nAll characters:', ''.join(chars))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Create mappings from characters to integers and back\n",
                "char_to_idx = {ch: i for i, ch in enumerate(chars)}\n",
                "idx_to_char = {i: ch for i, ch in enumerate(chars)}\n\n",
                "# Example: encode and decode some text\n",
                "example_text = \"Hello, World!\"\n",
                "encoded = [char_to_idx[ch] for ch in example_text]\n",
                "decoded = ''.join([idx_to_char[idx] for idx in encoded])\n\n",
                "print(f'Original text: {example_text}')\n",
                "print(f'Encoded: {encoded}')\n",
                "print(f'Decoded: {decoded}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Character Embeddings\n\nNow we'll create character embeddings. Each character will be represented by a vector in a higher-dimensional space."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Parameters for embeddings\n",
                "embedding_dim = 16  # Dimension of the embedding space\n\n",
                "# Create a random embedding table\n",
                "embeddings = torch.nn.Embedding(vocab_size, embedding_dim)\n\n",
                "# Example: get embeddings for a sequence of characters\n",
                "example_sequence = torch.tensor(encoded)  # Using the encoded text from above\n",
                "embedded_sequence = embeddings(example_sequence)\n\n",
                "print(f'Shape of embedded sequence: {embedded_sequence.shape}')\n",
                "print('\\nEmbedding for first character:')\n",
                "print(embedded_sequence[0].detach().numpy())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Creating Training Examples\n\nLet's see how to create training examples for a language model. Each example will consist of a sequence of characters and the next character as the target."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Convert entire text to indices\n",
                "data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)\n\n",
                "# Create sequences of context_length characters\n",
                "context_length = 8\n",
                "x = torch.stack([data[i:i+context_length] for i in range(len(data)-context_length)])\n",
                "y = data[context_length:]\n\n",
                "print(f'Shape of input sequences: {x.shape}')\n",
                "print(f'Shape of target values: {y.shape}')\n\n",
                "# Show an example\n",
                "idx = 0  # First sequence\n",
                "context = ''.join([idx_to_char[int(i)] for i in x[idx]])\n",
                "next_char = idx_to_char[int(y[idx])]\n",
                "print(f'\\nExample:')\n",
                "print(f'Context: \"{context}\"')\n",
                "print(f'Next character: \"{next_char}\"')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Visualizing Embeddings\n\nWe can visualize the learned embeddings by projecting them to 2D using PCA."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "from sklearn.decomposition import PCA\n",
                "import matplotlib.pyplot as plt\n\n",
                "# Get the embedding weights\n",
                "weights = embeddings.weight.detach().numpy()\n\n",
                "# Project to 2D using PCA\n",
                "pca = PCA(n_components=2)\n",
                "projected = pca.fit_transform(weights)\n\n",
                "# Plot\n",
                "plt.figure(figsize=(12, 8))\n",
                "plt.scatter(projected[:, 0], projected[:, 1], alpha=0.5)\n\n",
                "# Add character labels\n",
                "for i, char in enumerate(chars):\n",
                "    if char == '\\n': char = 'newline'  # Make newline visible\n",
                "    plt.annotate(char, (projected[i, 0], projected[i, 1]))\n\n",
                "plt.title('Character Embeddings Projected to 2D')\n",
                "plt.xlabel('First Principal Component')\n",
                "plt.ylabel('Second Principal Component')\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('notebooks/tokenization_and_embeddings.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 