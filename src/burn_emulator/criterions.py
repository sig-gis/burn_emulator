import torch
import torch.einsum as einsum
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class Reducer(nn.Module):
    def __init__(self,
                 reduction: str) -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        match self.reduction:
            case "none":
                return torch.identity(inputs)
            case "mean":
                return torch.mean(inputs)
            case "sum":
                return torch.sum(inputs)


class DiceLoss(nn.Module):
    def __init__(self,
                 reduction: str = 'mean',
                 multiclass: bool = False,
                 ignore_classes: Optional[list] = None,
                 epsilon: float = 1e-6,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
        self.multiclass = multiclass
        self.ignore_classes = ignore_classes
        self.epsilon = epsilon
        self.reducer = Reducer(reduction)
        
    def forward_loss(self, inpts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        intersection = einsum("bcwh,bcwh->bc", inpts, targets)
        sum_probs = einsum("bcwh->bc", inpts) + einsum("bcwh->bc", targets)
        loss = (2. * intersection + self.epsilon) / (sum_probs + self.epsilon)  

        return 1 - loss
    
    def forward_activation(self, inpts: torch.Tensor) -> torch.Tensor:
        if self.multiclass:
            inpts = F.softmax(inpts, dim=1)
        else:
            inpts = F.sigmoid(inpts)
        return inpts

    def forward(self, inpts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inpts = self.forward_activation(inpts)
        loss = self.forward_loss(inpts, targets)
        loss = self.reducer(loss)
        return loss