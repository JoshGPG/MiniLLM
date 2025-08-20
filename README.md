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

## Data Pipeline

The project includes a pipeline for collecting and preparing the training data. It runs in three stages:

1. **Fetch articles**

   Download raw reference material from the web and store it as JSON Lines:
   ```bash
   python src/data_pipeline.py fetch --output data/raw/articles.jsonl
   ```

2. **Generate Q&A pairs**

   Turn the articles into question/answer examples:
   ```bash
   python src/data_pipeline.py generate --input data/raw/articles.jsonl --output data/processed/qa_pairs.jsonl
   ```

3. **Create dataset splits**

   Produce train, validation, and test splits:
   ```bash
   python src/data_pipeline.py split --input data/processed/qa_pairs.jsonl --output-dir data/splits --train-size 0.8 --val-size 0.1 --test-size 0.1
   ```

## Data Directory

```
data/
├── raw/
│   └── articles.jsonl        # Raw articles; one JSON object per line
├── processed/
│   └── qa_pairs.jsonl        # Generated question/answer pairs
└── splits/
    ├── train.jsonl          # Training set (JSONL)
    ├── val.jsonl            # Validation set (JSONL)
    └── test.jsonl           # Test set (JSONL)
```

All files in `data/` use the [JSON Lines](https://jsonlines.org/) format.

