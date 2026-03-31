from tokenizers import Tokenizer
import argparse
from nanomoe.data import helpers

def infer(tokenizer_path,data_path,save_to_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(tokenizer.get_vocab_size())

    train_data, val_data = helpers.train_test_split(data_path)
    train_ids = helpers.encode_data(
        tokenizer=tokenizer,
        data=train_data
    )
    val_ids = helpers.encode_data(
        tokenizer=tokenizer,
        data=val_data
    )

    helpers.save_ids(
        train_ids=train_ids,
        val_ids=val_ids,
        output_path=save_to_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", help="Path to the tokenizer")
    parser.add_argument("--data", help="Path to the .txt file")
    parser.add_argument("--save_to", help="Path to the folder where .bin are saved")
    args = parser.parse_args()

    infer(
        tokenizer_path=args.tokenizer,
        data_path=args.data,
        save_to_path=args.save_to
    )
