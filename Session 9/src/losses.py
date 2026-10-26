import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def make_targets(tokens, pad_id, horizon=1, ignore_index=IGNORE_INDEX):
    """Targets at t+horizon; padding targets are ignored by cross entropy."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    targets = tokens[:, horizon:].clone()
    targets[targets == pad_id] = ignore_index
    return targets


def next_token_loss(logits, tokens, pad_id, ignore_index=IGNORE_INDEX):
    """Align logits at t with token targets at t+1 and ignore padding."""
    targets = make_targets(tokens, pad_id, horizon=1, ignore_index=ignore_index)
    aligned_logits = logits[:, :-1, :]
    if aligned_logits.shape[:2] != targets.shape:
        raise ValueError(f"Alignment mismatch: logits {tuple(aligned_logits.shape)} targets {tuple(targets.shape)}")
    loss = F.cross_entropy(aligned_logits.reshape(-1, aligned_logits.shape[-1]),
                           targets.reshape(-1), ignore_index=ignore_index)
    return loss, targets


def horizon_loss(head, hidden, tokens, pad_id, horizon, ignore_index=IGNORE_INDEX):
    """Loss for a head predicting token t+horizon from hidden state at t."""
    aligned_hidden = hidden[:, :-horizon, :]
    targets = make_targets(tokens, pad_id, horizon=horizon, ignore_index=ignore_index)
    logits = head(aligned_hidden)
    if logits.shape[:2] != targets.shape:
        raise ValueError(f"Alignment mismatch: logits {tuple(logits.shape)} targets {tuple(targets.shape)}")
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=ignore_index)
    return loss, targets


def chunked_cross_entropy(hidden, head, targets, chunk_size=32, ignore_index=IGNORE_INDEX):
    """Chunk positions to limit logits memory; average by valid-token count."""
    total_loss = hidden.new_zeros(())
    valid_count = int((targets != ignore_index).sum().item())
    if valid_count == 0:
        raise ValueError("No valid targets: loss is undefined")
    for start in range(0, hidden.shape[1], chunk_size):
        end = min(start + chunk_size, hidden.shape[1])
        chunk_logits = head(hidden[:, start:end, :])
        chunk_targets = targets[:, start:end]
        total_loss = total_loss + F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.shape[-1]), chunk_targets.reshape(-1),
            ignore_index=ignore_index, reduction="sum")
    return total_loss / valid_count
