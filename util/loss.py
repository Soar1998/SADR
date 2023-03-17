import torch
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict


def square_loss(y_pre, y_true, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    if isinstance(y_true, (float, int)):
        y_true = y_pre.new_full(y_pre.size(), y_true)

    loss = F.mse_loss(input=y_pre, target=y_true, reduction="none")
    return _reduce_loss(loss, reduction)


def sigmoid_cross_entropy(y_pre, y_true, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    loss = F.binary_cross_entropy_with_logits(input=y_pre, target=y_true, reduction="none")
    return _reduce_loss(loss, reduction)


@typeassert(loss=str, reduction=str)
def pointwise_loss(loss, y_pre, y_true, reduction=Reduction.SUM):
    Reduction.validate(reduction)

    losses = OrderedDict()
    losses["square"] = square_loss
    losses["sigmoid_cross_entropy"] = sigmoid_cross_entropy

    if loss not in losses:
        loss_list = ', '.join(losses.keys())
        ValueError(f"'loss' is invalid, and must be one of '{loss_list}'")

    return losses[loss](y_pre, y_true, reduction=reduction)


def bpr_loss(y_diff, reduction=Reduction.SUM):
    """bpr loss
    """
    Reduction.validate(reduction)
    loss = -F.logsigmoid(y_diff)
    # loss = F.softplus(-y_diff)
    return _reduce_loss(loss, reduction)


def hinge(y_diff, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    ones = y_diff.new_ones(y_diff.size())
    loss = torch.relu(ones-y_diff)
    return _reduce_loss(loss, reduction)


@typeassert(loss=str, reduction=str)
def pairwise_loss(loss, y_diff, reduction=Reduction.SUM):
    Reduction.validate(reduction)

    losses = OrderedDict()
    losses["bpr"] = bpr_loss
    losses["hinge"] = hinge
    losses["square"] = partial(square_loss, y_true=1.0)

    if loss not in losses:
        loss_list = ', '.join(losses.keys())
        ValueError(f"'loss' is invalid, and must be one of '{loss_list}'")

    return losses[loss](y_diff, reduction=reduction)


def l2_loss(*weights):
    """L2 loss

    Compute  the L2 norm of tensors without the `sqrt`:

        output = sum([sum(w ** 2) / 2 for w in weights])

    Args:
        *weights: Variable length weight list.

    """
    loss = 0.0
    for w in weights:
        loss += torch.sum(torch.pow(w, 2))

    return 0.5*loss
