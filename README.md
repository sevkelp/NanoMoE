# NanoMoE
Small Transformer where the Feed-Forward Network (FFN) is replaced by a Router and 4 tiny "Expert" networks.

# Commands
## Tokenizer
We ended up not using the custom tokenizer (below are examples to run it).
- Train tokenizer : `src/nanomoe/tokenizer/train.py --model_config configs/gpt.yaml --train_data data/toy_data.txt --save_to notebooks/tokenizer.json`
- Infer tokenizer : `src/nanomoe/tokenizer/infer.py --tokenizer notebooks/tokenizer.json --data data/toy_data.txt --save_to data/custom`
## Training
`train-m1-gpt2`
## Inference
```python
from nanomoe.model import GPT

# Single-line loading from artifact folder
model, config = GPT.from_pretrained(path_to_checkpoint_folder)
model.eval()

# Generate text
output = model.generate('ROMEO:', max_new_tokens=100)
print(output)
```
# Current Results
- Validation loss : 3.4
- Training loss : 2.74
- Perplexity : 30

# Project structure
├── configs/           # YAML configuration files for models and training
├── nanomoe/
│   ├── model.py       # GPT, Transformer Block, and Attention logic
│   ├── training.py    # LightningModule wrapper
│   ├── config.py      # Pydantic schemas for type-safe configuration
│   └── data/          # Data loading and tokenization utilities
├── scripts/           # Training and inference entry points
└── Makefile           # Shortcuts for common commands

# TODO
Update model architecture and replace MLP with MoE.
