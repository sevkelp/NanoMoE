import os
import requests
import numpy as np
import tiktoken

def dload_toy_data(input_file_path: str, output_path: str):
    # Download if doesn't exist
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    if not os.path.exists(os.path.join(output_path, 'train.bin')):
        train_ids = np.array(train_ids, dtype=np.uint16)
        train_ids.tofile(os.path.join(output_path, 'train.bin'))

    if not os.path.exists(os.path.join(output_path, 'val.bin')):
        val_ids = np.array(val_ids, dtype=np.uint16)
        val_ids.tofile(os.path.join(output_path, 'val.bin'))
