train-m1-custom:
	uv run scripts/train.py --train-data data/custom/train.bin --val-data data/custom/val.bin --model configs/gpt.yaml --trainer configs/train_mac.yaml
train-m1-gpt2:
	uv run scripts/train.py --train-data data/gpt2/train.bin --val-data data/gpt2/val.bin --model configs/gpt.yaml --trainer configs/train_mac.yaml
