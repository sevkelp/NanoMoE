import os
import math
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension,dimension)
        self.linear_2 = nn.Linear(dimension,dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)

        return swiglu


class AttentionHead(nn.Module):
    """
    For now number of heads is 1 meaning the entire input is ingested in 1 head.
    When it will be > 1 we'll partition the input so that each chunk is ingested by a head.
    """
    def __init__(self, n_embed, attention = "flash", block_size = None):
        super().__init__()
        # self.block_size = block_size
        self.n_embed = n_embed
        self.attention = attention
        self.block_size = block_size
        self.Q = nn.Linear(self.n_embed, self.n_embed)
        self.K = nn.Linear(self.n_embed, self.n_embed)
        self.V = nn.Linear(self.n_embed, self.n_embed)
        if (self.attention != "flash") and (self.block_size == None):
            raise Exception("`block_size` required for regular attention, otherwise use `attention = `flash``")
        if self.block_size is not None:
            self.register_buffer("mask", torch.tril(torch.ones((1,self.block_size,self.block_size))) == 0)
            #self.mask = torch.tril(torch.ones((1,self.block_size,self.block_size))) == 0

    def forward(self, X):
        # X -> B,T,C
        B,T,C = X.shape
        q = self.Q(X) # B,T,n_embed @ n_embed,n_embed -> B,T,n_embed
        k = self.K(X) # B,T,n_embed @ n_embed,n_embed -> B,T,n_embed
        v = self.V(X) # B,T,n_embed @ n_embed,n_embed -> B,T,n_embed

        # Using flash attention
        if self.attention == "flash":
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        else :
            attention = q @ k.transpose(-2,-1) * math.sqrt(1.0/k.size(-1)) # B,T,T
            attention = attention.masked_fill(self.mask, float('-inf')) # most likely self.mask[:,:T,:T]
            attention = F.softmax(attention, dim=-1)
            dropout = nn.Dropout(p=0.2)
            attention = dropout(attention)
            y = attention @ v # B,T,T @ B,T,n_embed -> B,T,n_embed

        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_heads, attention="flash", block_size=None):
        super().__init__()
        assert n_embed % n_heads == 0, "n_embed must be divisible by n_heads"

        self.n_embed = n_embed
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.attention = attention
        self.dropout = nn.Dropout(p=0.2)

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed)
        # Final output projection to mix the heads
        self.o_proj = nn.Linear(n_embed, n_embed)

        if (self.attention != "flash") and (block_size is None):
            raise Exception("`block_size` required for regular attention")

        if block_size is not None:
            # Mask remains the same size, but we'll apply it across all heads
            self.register_buffer("mask", torch.tril(torch.ones((block_size, block_size))) == 0)

    def forward(self, x):
        B, T, C = x.shape # Batch, Time, Channels (n_embed)

        # 1. Project to Q, K, V in one go and split
        qkv = self.qkv_proj(x) # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embed, dim=2)

        # 2. Reshape for Multi-Head: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        # Moving n_heads to dim 1 allows batch matrix multiplication across T
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Attention logic
        if self.attention == "flash":
            # PyTorch's SDPA handles the multi-head dimension automatically
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        else:
            # Manual scaled dot-product attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:T, :T], float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v # (B, n_heads, T, head_dim)

        # 4. Concatenate heads back together: (B, T, n_embed)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Final projection (mixing the independent head information)
        return self.dropout(self.o_proj(y))

class MLP(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.n_embed = n_embed
        self.op1 = nn.Linear(self.n_embed, 4 * self.n_embed)
        self.o_proj = nn.Linear(4 * self.n_embed, self.n_embed)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.op1(x)
        x = self.gelu(x)
        x = self.o_proj(x)
        x = self.dropout(x)
        return x

class Transformer(nn.Module):
    def __init__(self, n_embed, n_heads, attention = "flash", block_size = None):
        super().__init__()
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.attention = attention
        self.block_size = block_size
        #self.pos_embed = nn.Embedding(block_size, n_embed)
        self.att = MultiHeadAttention(
            n_embed=self.n_embed,
            n_heads=n_heads,
            attention=self.attention,
            block_size=self.block_size
        )
        self.mlp = MLP(n_embed=self.n_embed)
        self.rms1 = nn.RMSNorm([self.n_embed])
        self.rms2 = nn.RMSNorm([self.n_embed])

    def forward(self, x):
        x = x + self.att(self.rms1(x))
        x = x + self.mlp(self.rms2(x))
        return x

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.o_proj = nn.Linear(self.input_size, self.num_classes)

    def forward(self, x):
        return self.o_proj(x)

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embed, n_heads, n_layers, attention = "flash", block_size = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention = attention
        self.block_size = block_size
        self.embed = torch.nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(block_size, n_embed)
        self.rms = nn.RMSNorm([self.n_embed])
        # Stacking the blocks
        self.blocks = nn.ModuleList([
            Transformer(
                n_embed=self.n_embed,
                n_heads=n_heads,
                attention=self.attention,
                block_size=self.block_size
            )
            for _ in range(n_layers)
        ])
        self.classif = Classifier(
            input_size=self.n_embed,
            num_classes=self.vocab_size
        )

        # Weight initialization
        self.apply(self._init_weights)
        for p_name,p in self.named_parameters():
            if "o_proj.weight" in p_name:
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))

        # Weight Tying
        self.classif.o_proj.weight = self.embed.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # The "Gold Standard" for GPT
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.embed(x)#*torch.sqrt(torch.tensor(self.n_embed))
        # Positional embedding
        device = x.device
        B, T, C = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=device) # Match length T
        x = x + self.pos_embed(pos).unsqueeze(0)
        # Blocks
        for block in self.blocks:
            x = block(x)
        x = self.classif(self.rms(x))
        return x

    def get_num_param(self):
        param_num = 0
        for _, p in self.named_parameters():
            param_num += p.numel()
        return param_num

    @classmethod
    def from_pretrained(cls, folder_path: str | Path):
        folder_path = Path(folder_path)

        # 1. Load the Config
        from nanomoe.config import GPTConfig
        with open(os.path.join(folder_path, "config.json"),"r") as f:
            config_data = json.load(f)
        config = GPTConfig.from_dict(config_data)

        # 2. Re-initialize the model architecture
        model = cls(**config.model_dump())

        # 3. Load the weights
        state_dict = torch.load(folder_path / "model_weights.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        return model, config
