import torch
import torch.nn.functional as F

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Applies a masked softmax to the logits.
    
    Args:
        logits: Raw tensor logits.
        mask: Boolean action mask where True means the action is valid.
        dim: The dimension along which softmax will be computed.
        
    Returns:
        A tensor of probabilities where masked actions have 0.0 probability.
    """
    # Replace False (invalid) with -1e9
    masked_logits = logits.masked_fill(~mask, -1e9)
    return F.softmax(masked_logits, dim=dim)
