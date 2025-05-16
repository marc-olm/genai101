#  <img src="docs/images/genai101-logo.png" alt="GenAI 101 Logo" width="70" align="center"/> GenAI 101: Intro to GenAI and LLMs

This repository contains materials and code examples for a hands-on course on getting started with Generative AI and Large Language Models (LLMs). The course is designed to help participants understand the fundamentals of GenAI and gain practical experience working with LLMs.

## 📒 Running notebooks in Google Colab 

- The notebooks and dependencies are self-contained to be runnable from a Google Colab. 
- In general, open [Google Colab](https://colab.research.google.com/) to run the code directly in your browser. Choose GitHub option, enter the current GitHub URL https://github.com/marc-olm/genai101, search and select the notebook. 

- The following direct links to relevant notebooks are also provided: 

| Notebook | Open in Colab |
|----------|----------------|
| Language Modelling 101 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marc-olm/genai101/blob/main/notebooks/tokenization_and_embeddings.ipynb) |
| Your first RAG pipeline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marc-olm/genai101/blob/main/notebooks/rag_playground.ipynb) |


## 💻 Local setup

Alternatively, you can setup your local environment in your macOS machine, clone the repo and run locally:

```bash
# Clone the repository
git clone https://github.com/marc-olm/genai101.git
cd genai101

# Run the setup script
./setup.sh
```

This script will:
1. Create a virtual environment
2. Install all required dependencies
3. Set up the Jupyter kernel
4. Configure everything you need to run the notebooks

This course uses local open-source LLMs via [ollama](https://ollama.com/) for inference and embedding tasks.

## Project Structure

- `notebooks/`: Contains Jupyter notebooks with the course materials
- `data/`: Contains the Shakespeare dataset
- `requirements.txt`: Lists all Python dependencies
- `setup.sh`: Setup script to configure the environment

## References 

- RNNs parts are inspired by [rnn-effectiveness](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) and code adapted from by [github.com/karpathy/char-rnn](https://github.com/karpathy/char-rnn) by Andrej Karpathy


## License

This project is licensed under the [MIT License](license)