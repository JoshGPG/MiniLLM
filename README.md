# MiniLLM

MiniLLM is a beginner-friendly large language model project built to explore the fundamentals of how LLMs work under the hood. The goal of this project isn't to compete with state-of-the-art models, but to create a small-scale system for learning, experimenting, and applying core concepts in natural language processing.

## Project Scope
MiniLLM focuses on straightforward factual domains such as astronomy, human anatomy, animals, geography, basic chemistry, and weather phenomena. The model is expected to produce concise answers, typically 2–4 sentences.

## Environment Setup
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Repository Structure
```
mini_llm/
├── data/
├── src/
│   ├── tokenizer.py
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   └── utils.py
├── experiments/
└── README.md
```
