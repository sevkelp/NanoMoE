import os
import requests
import numpy as np
import tiktoken

def train_test_split(input_file_path:str):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    return train_data, val_data

def encode_data(tokenizer, data):
    # Definitely should handle this with classes rather than strings
    if "tiktoken" in str(type(tokenizer)):
        print("Using GPT2 tokenizer")
        data_ids = tokenizer.encode_ordinary(data)
    elif "tokenizers" in str(type(tokenizer)):
        print("Using custom tokenizer")
        data_ids = tokenizer.encode(data).ids
    return data_ids

def save_ids(train_ids, val_ids, output_path, overwrite = True):
    if (not os.path.exists(os.path.join(output_path, 'train.bin'))) or overwrite:
        train_ids = np.array(train_ids, dtype=np.uint16)
        train_ids.tofile(os.path.join(output_path, 'train.bin'))

    if (not os.path.exists(os.path.join(output_path, 'val.bin'))) or overwrite:
        val_ids = np.array(val_ids, dtype=np.uint16)
        val_ids.tofile(os.path.join(output_path, 'val.bin'))


def dload_toy_data(input_file_path: str, output_path: str):
    # Download if doesn't exist
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)

    train_data, val_data = train_test_split(input_file_path=input_file_path)

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = encode_data(enc, data=train_data)
    val_ids = encode_data(enc, data=val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    save_ids(train_ids=train_ids, val_ids=val_ids, output_path=output_path)
