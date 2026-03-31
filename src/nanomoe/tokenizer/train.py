from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from nanomoe import config
from nanomoe.data import helpers
import argparse

def train(vocab_size, train_data, save_to_path = None):
    # 1. Initialize a BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # 2. Configure the trainer
    # Set your target vocab_size here!
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    # 3. Train it on your text files
    tokenizer.train_from_iterator([train_data], trainer)
    print(tokenizer.get_vocab_size())

    # 4. Save it
    if save_to_path is not None:
        tokenizer.save(save_to_path)
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", help="path to the training config")
    parser.add_argument("--train_data", help="path to the training data")
    parser.add_argument("--save_to", help="Where to drop the trained tokenizer")
    args = parser.parse_args()

    model_config = config.GPTConfig.from_yaml(args.model_config)
    train_data, _ = helpers.train_test_split(args.train_data)

    train(
        vocab_size=model_config.vocab_size,
        train_data=train_data,
        save_to_path=args.save_to
    )
