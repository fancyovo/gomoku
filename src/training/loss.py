import math
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
    value_logits: torch.Tensor,
    value_targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    value_weights: torch.Tensor | None = None,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    normalize: bool = True,
):
    """AlphaGo Zero loss. Set pw=0 or vw=0 for alternating optimizer steps.

    Args:
        policy_logits:  (N, n_positions) — model policy head outputs (logits)
        mcts_targets:   (N, n_positions) — MCTS visit count distribution (target)
        value_logits:   (N, 2) — model value head raw logits [win, lose]
        value_targets:  (N, 2) — soft targets: [1,0]=win, [0,1]=lose, [.5,.5]=draw
        mask:           (N,) — True for valid positions
        value_weights:  (N,) — per-sample weight for value loss (e.g. decay by dist to end)
    """
    if mask is not None:
        policy_logits = policy_logits[mask]
        mcts_targets = mcts_targets[mask]
        value_logits = value_logits[mask]
        value_targets = value_targets[mask]
        if value_weights is not None:
            value_weights = value_weights[mask]

    if mask is None or mask.sum() == 0:
        return (torch.tensor(0.0, device=policy_logits.device),
                torch.tensor(0.0, device=policy_logits.device),
                torch.tensor(0.0, device=policy_logits.device))

    # Policy: CE over n_policy classes
    n_policy = policy_logits.size(-1)  # 225
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_ce_per_sample = -(mcts_targets * policy_log_probs).sum(dim=-1)  # (N,)
    if value_weights is not None:
        W = value_weights.sum()
        policy_loss = (policy_ce_per_sample * value_weights).sum() / W
    else:
        policy_loss = policy_ce_per_sample.mean()
    if normalize:
        policy_loss = policy_loss / (math.log(n_policy) if n_policy > 1 else 1.0)

    # Value: CE over n_value classes
    n_value = value_logits.size(-1)  # 2
    value_log_probs = F.log_softmax(value_logits.float(), dim=-1)
    value_ce_per_sample = -(value_targets * value_log_probs).sum(dim=-1)  # (N,)
    if value_weights is not None:
        value_loss = (value_ce_per_sample * value_weights).sum() / W
    else:
        value_loss = value_ce_per_sample.mean()
    if normalize:
        value_loss = value_loss / (math.log(n_value) if n_value > 1 else 1.0)

    loss = policy_loss * policy_weight + value_loss * value_weight
    return loss, policy_loss.detach(), value_loss.detach()
