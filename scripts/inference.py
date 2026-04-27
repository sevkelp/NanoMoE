# inference.py
import torch
import tiktoken
from nanomoe.model import GPT

def generate(model_path, prompt="ROMEO:", max_new_tokens=50):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # The "Clean" way to load
    model, config = GPT.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Simple generation loop
    tokenizer = tiktoken.get_encoding("gpt2")
    x = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop x if it exceeds block_size
            x_cond = x[:, -config.block_size:]
            logits = model(x_cond)
            logits = logits[:, -1, :] # focus only on the last time step

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

    return tokenizer.decode(x[0].tolist())

if __name__ == "__main__":
    print(generate("checkpoints/run_001"))
