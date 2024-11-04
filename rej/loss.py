from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPELoss(ABC):
    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return self.pw_risk(y, y_hat)

    @abstractmethod
    def pw_risk(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        pass

    def risk(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return self.pw_risk(y, y_hat).mean()

    @abstractmethod
    def pw_bayes_risk(self, y: torch.Tensor) -> torch.Tensor:
        pass


class ProperCPELoss(CPELoss):
    def pw_bayes_risk(self, y: torch.Tensor) -> torch.Tensor:
        return self.pw_risk(y, y)


class LogLoss(ProperCPELoss):
    def pw_risk(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return -(y * y_hat.log()).sum(dim=1)

    def pw_logit_risk(
        self, y: torch.Tensor, log_y_hat: torch.Tensor
    ) -> torch.Tensor:
        return -(y * F.log_softmax(log_y_hat, dim=1)).sum(dim=1)


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(
                bin_upper.item()
            )
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += (
                    torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                    * prop_in_bin
                )

        return ece


def logistic(z: torch.Tensor) -> torch.Tensor:
    return torch.where(-z < 50, torch.log1p(torch.exp(-z)), -z)
    # torch.log(1 + torch.exp(-z) + 1e-10)
