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


def alphago_zero_loss(
    policy_logits: torch.Tensor,
    mcts_targets: torch.Tensor,
    value_preds: torch.Tensor,
    value_targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    value_weight: float = 1.0,
):
    """AlphaGo Zero loss: cross-entropy on policy + MSE on value.

    Args:
        policy_logits:  (N, n_positions) — model policy head outputs (logits)
        mcts_targets:   (N, n_positions) — MCTS visit count distribution (target)
        value_preds:    (N,) — model value head outputs (tanh, -1..1)
        value_targets:  (N,) — game outcome from current player's perspective (+1/-1/0)
        mask:           (N,) — True for valid positions
        value_weight:   float — weight for value loss relative to policy loss
    """
    if mask is not None:
        policy_logits = policy_logits[mask]
        mcts_targets = mcts_targets[mask]
        value_preds = value_preds[mask]
        value_targets = value_targets[mask]

    if len(value_targets) == 0:
        return (torch.tensor(0.0, device=policy_logits.device),
                torch.tensor(0.0, device=policy_logits.device),
                torch.tensor(0.0, device=policy_logits.device))

    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(mcts_targets * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(value_preds, value_targets)

    loss = policy_loss + value_weight * value_loss
    return loss, policy_loss.detach(), value_loss.detach()
