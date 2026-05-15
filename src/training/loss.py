import torch
import torch.nn.functional as F


def reinforce_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    entropy_coef: float = 0.01,
):
    """
    REINFORCE policy gradient loss with entropy bonus.

    Args:
        logits:       (n_moves, n_positions) — model output logits
        actions:      (n_moves,) — chosen action indices
        rewards:      (n_moves,) — +1/-1 terminal reward per move
        entropy_coef: weight for entropy bonus

    Returns:
        (loss, policy_loss, entropy) tuple
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)

    # Policy gradient: -log_prob * reward
    selected_log_probs = log_probs[torch.arange(len(actions)), actions]
    policy_loss = -(selected_log_probs * rewards).mean()

    # Entropy bonus: encourage exploration
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    entropy_loss = -entropy_coef * entropy

    loss = policy_loss + entropy_loss
    return loss, policy_loss.detach(), entropy.detach()
