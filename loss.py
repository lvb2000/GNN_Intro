import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(pred, true):
    """
    Compute loss and prediction score

    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Grou

    Returns: Loss, normalized prediction score

    """
    bce_loss = nn.BCEWithLogitsLoss()

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    # CrossEntropy Loss
    # multiclass
    if pred.ndim > 1 and true.ndim == 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true), pred
    # binary or multilabel
    else:
        true = true.float()
        return bce_loss(pred, true), torch.sigmoid(pred)