# NanoMoE
Small Transformer where the Feed-Forward Network (FFN) is replaced by a Router and 4 tiny "Expert" networks.

### Commands
- Train tokenizer : `src/nanomoe/tokenizer/train.py --model_config configs/gpt.yaml --train_data data/toy_data.txt --save_to notebooks/tokenizer.json`
- Infer tokenizer : `src/nanomoe/tokenizer/infer.py --tokenizer notebooks/tokenizer.json --data data/toy_data.txt --save_to data/custom`
