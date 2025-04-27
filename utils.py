from typing import List
from torch import Tensor
from torch import optim
from torch.optim import Optimizer
import math

def lexsort(
    keys: List[Tensor],
    dim: int = -1,
    descending: bool = False,
) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """
    assert len(keys) >= 1

    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)
    return out

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)