# Experiments Log

## Datasets and Preprocessing
- Used a toy question–answer dataset stored as JSON in `data/splits`.
- Combined train and validation texts to build a byte-pair encoding vocabulary with a maximum size of 8k tokens.
- Each example tokenised with BOS and EOS tokens; padded to build inputs and shifted targets.

## Model Architectures and Hyperparameters
- Base model: 2-layer Transformer with 32-dim embeddings, 2 attention heads and 64-dim feed-forward blocks.
- Experiments ran for 2 epochs each with Adam (lr=1e-3) and batch size 16.
- Training curves (`experiments/exp1/log.jsonl`, `experiments/exp2/log.jsonl`) show train loss decreasing from ~4.6 to ~3.3 while validation loss stagnates around 4.1–4.7.

## Observations
- Rapid drop in training loss but little improvement in validation loss indicates overfitting to the tiny dataset.
- Generated answers often repeat tokens or echo questions, showing limited generalisation.

## Challenges
- High sensitivity to learning rate; lower values slowed convergence while higher values diverged.
- Tokeniser occasionally produced unknown tokens for rare words, suggesting limited vocabulary coverage.
- Dataset size (fewer than 10 QA pairs) provides poor coverage and leads to unstable validation metrics.

## Improvements for Future Work
- Gather a larger and more diverse QA corpus.
- Apply regularisation: dropout, weight decay or label smoothing.
- Implement causal attention masks and learning rate schedulers for more stable optimisation.

## Potential Next Steps
- Explore retrieval-augmented generation to incorporate external facts.
- Use beam search or nucleus sampling during generation for more diverse answers.
- Train with additional tasks (e.g., summarisation or translation) to encourage better language modelling.

