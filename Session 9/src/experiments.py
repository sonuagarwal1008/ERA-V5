import math, gc, time, threading, os
import torch
import torch.nn.functional as F
from .losses import IGNORE_INDEX, make_targets, chunked_cross_entropy


def decode_token(tokenizer, token_id, pad_id):
    if int(token_id) == pad_id:
        return "<PAD/EOT>"
    value = tokenizer.decode([int(token_id)])
    # repr makes leading spaces visible (GPT-2 BPE tokens often contain one).
    return repr(value)[1:-1] if value.strip() != value else value


def print_shapes(tokens, hidden, logits):
    b, t = tokens.shape
    print("Tensor shapes (B=batch, T=sequence positions, D=model width, V=vocabulary):")
    items = [
        ("tokens", tuple(tokens.shape), "B sequences x T token IDs"),
        ("hidden", tuple(hidden.shape), "B x T x D contextual vectors"),
        ("logits", tuple(logits.shape), "B x T x V scores for vocabulary tokens"),
        ("logits[:, :-1]", tuple(logits[:, :-1].shape), "B x (T-1) x V; last position has no next target"),
        ("tokens[:, 1:]", tuple(tokens[:, 1:].shape), "B x (T-1); target is shifted one position forward"),
        ("flat logits", (b*(t-1), logits.shape[-1]), "N x V; N=B*(T-1) prediction rows"),
        ("flat targets", (b*(t-1),), "N target IDs, one per prediction row"),
    ]
    for name, shape, meaning in items:
        print(f"  {name:18} {str(shape):20} {meaning}")


def print_alignment(tokens, tokenizer, pad_id, rows=12, horizon=2):
    print("\nDecoded token alignment (strings, not IDs):")
    print(f"{'position':>8}  {'input':<18} {'target t+1':<18} {'target t+2':<18}")
    limit = min(rows, tokens.shape[1]-horizon)
    for i in range(limit):
        a = decode_token(tokenizer, tokens[0, i], pad_id)
        b = decode_token(tokenizer, tokens[0, i+1], pad_id)
        c = decode_token(tokenizer, tokens[0, i+2], pad_id)
        print(f"{i:8d}  {a:<18} {b:<18} {c:<18}")


def measure_parameter_tying(model_factory, vocab_size, d_model, device):
    untied = model_factory(vocab_size=vocab_size, d_model=d_model, tie_weights=False).to(device)
    tied = model_factory(vocab_size=vocab_size, d_model=d_model, tie_weights=True).to(device)
    p_u = sum(p.numel() for p in untied.parameters())
    p_t = sum(p.numel() for p in tied.parameters())
    expected = vocab_size*d_model
    print(f"Untied parameters: {p_u:,}")
    print(f"Tied parameters:   {p_t:,}")
    print(f"Saved:             {p_u-p_t:,} (expected V*D={expected:,})")
    print(f"Parameter-count check: {'PASS' if p_u-p_t == expected else 'FAIL'}")
    return p_u, p_t


def benchmark_memory(hidden, head, targets, chunk_size=32, device="cpu"):
    """GPU peak allocation is measured accurately; CPU RSS is labelled noisy."""
    # This benchmark compares forward/inference peak memory. For training with
    # gradients, ordinary chunk accumulation can retain each chunk's graph;
    # use per-chunk backward or a custom autograd function for a true training
    # memory reduction.
    def ordinary():
        with torch.no_grad():
            logits = head(hidden)
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                   ignore_index=IGNORE_INDEX)
    def chunked():
        with torch.no_grad():
            return chunked_cross_entropy(hidden, head, targets, chunk_size=chunk_size)
    if device.startswith("cuda"):
        results = {}
        for name, fn in [("ordinary", ordinary), ("chunked", chunked)]:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            loss = fn()
            torch.cuda.synchronize()
            results[name] = (float(loss.item()), torch.cuda.max_memory_allocated()/1024**2)
        print("GPU forward-only peak allocated memory (MiB):")
        for name, (loss, peak) in results.items(): print(f"  {name:9} loss={loss:.6f}, peak={peak:.2f} MiB")
        print(f"  ordinary/chunked peak ratio: {results['ordinary'][1]/max(results['chunked'][1],1e-9):.2f}x")
        return results
    # CPU RSS is process-wide and allocator-dependent; do not claim it is a clean peak-tensor measure.
    ordinary_loss, chunk_loss = float(ordinary().item()), float(chunked().item())
    full_mb = hidden.shape[0]*hidden.shape[1]*head.out_features*hidden.element_size()/1024**2
    chunk_mb = hidden.shape[0]*min(chunk_size,hidden.shape[1])*head.out_features*hidden.element_size()/1024**2
    print("CPU forward-only run: analytic logits-memory estimate (RSS is allocator-dependent):")
    print(f"  ordinary logits tensor: {full_mb:.2f} MiB")
    print(f"  chunked logits tensor:  {chunk_mb:.2f} MiB")
    print(f"  analytic ratio:         {full_mb/max(chunk_mb,1e-9):.2f}x")
    print(f"  losses: ordinary={ordinary_loss:.6f}, chunked={chunk_loss:.6f}, abs diff={abs(ordinary_loss-chunk_loss):.3e}")
    return {"ordinary_loss": ordinary_loss, "chunked_loss": chunk_loss,
            "ordinary_logits_mib": full_mb, "chunked_logits_mib": chunk_mb}
