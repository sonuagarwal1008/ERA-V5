import torch
import torch.nn as nn

class CausalLanguageModel(nn.Module):
    """Small causal Transformer with one or two independent vocabulary heads."""
    def __init__(self, vocab_size, d_model=128, n_layer=2, n_head=4,
                 max_len=512, two_heads=False, tie_weights=False):
        super().__init__()
        if d_model % n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.vocab_size, self.d_model = vocab_size, d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4*d_model,
            batch_first=True, norm_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layer, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.head1 = nn.Linear(d_model, vocab_size, bias=False)
        self.head2 = nn.Linear(d_model, vocab_size, bias=False) if two_heads else None
        if tie_weights:
            self.head1.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        if not tie_weights:
            nn.init.normal_(self.head1.weight, mean=0.0, std=0.02)
        if self.head2 is not None:
            nn.init.normal_(self.head2.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [B,T], got {tuple(tokens.shape)}")
        batch_size, seq_len = tokens.shape
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError("Sequence length exceeds configured max_len")
        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        # True means 'do not attend here': block future positions.
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=tokens.device,
                                             dtype=torch.bool), diagonal=1)
        return self.final_norm(self.transformer(hidden, mask=causal_mask))

    def output_heads(self):
        return ["head1"] + (["head2"] if self.head2 is not None else [])


def count_parameters(model):
    """Count unique parameters (tied weights are counted once)."""
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p)); total += p.numel()
    return total
