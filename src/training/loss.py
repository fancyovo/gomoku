import torch
import torch.nn.functional as F


def reinforce_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor | None = None,
    entropy_coef: float = 0.01,
    loss_scale: float = 1.0,
    action_mask: torch.Tensor | None = None,
):
    """
    REINFORCE policy gradient loss with entropy bonus.

    Args:
        logits:       (N, n_positions) — model output logits
        actions:      (N,) — chosen action indices
        rewards:      (N,) — +1/-1/0 terminal reward per move
        mask:         (N,) — True for valid positions (non-padding, reward may be 0 for
                           padding after game end, but those are excluded by mask)
        entropy_coef: weight for entropy bonus
        action_mask:  (N, n_positions) — True for occupied (illegal) positions

    Returns:
        (loss, policy_loss, entropy) tuple
    """
    if mask is not None:
        logits = logits[mask]
        actions = actions[mask]
        rewards = rewards[mask]
        if action_mask is not None:
            action_mask = action_mask[mask]

    if len(actions) == 0:
        return torch.tensor(0.0, device=logits.device), \
               torch.tensor(0.0, device=logits.device), \
               torch.tensor(0.0, device=logits.device)

    if action_mask is not None:
        logits = logits.masked_fill(action_mask, -1e9)

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)

    selected_log_probs = log_probs[torch.arange(len(actions), device=actions.device), actions]
    policy_loss = -(selected_log_probs * rewards).mean()

    # 0 * (-inf) = NaN from masked positions; nan_to_num fixes this
    entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1).mean()
    entropy_loss = -entropy_coef * entropy

    loss = policy_loss * loss_scale + entropy_loss
    return loss, policy_loss.detach(), entropy.detach()
