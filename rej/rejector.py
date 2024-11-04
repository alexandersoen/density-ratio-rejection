import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import bisect

from rej.classifier import CPEClassifier
from rej.div import AlphaDiv, KLDiv
from rej.loss import CPELoss


class RejectorNotFitted(Exception):
    pass


class Rejector(ABC, nn.Module):
    def __call__(self, x: torch.Tensor, tau: float):
        return self.rejection(x, tau)

    @abstractmethod
    def rejection(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        pass

    @abstractmethod
    def fit(self, train_dataloader: torch.utils.data.DataLoader, kwargs: dict[Any, Any] = {}) -> None:
        pass

    @property
    @abstractmethod
    def save_name(self) -> str:
        pass

    def save_path(self, base_path: Path | str, subdir: Path | str) -> Path:
        path = Path(base_path, subdir, f"{self.save_name}.pt")
        return path

    def save(self, base_path: Path | str, subdir: Path | str):
        path = Path(base_path, subdir, f"{self.save_name}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, base_path: Path | str, subdir: Path | str, device: torch.device=torch.device('cpu'), force_ordering=False) -> None:
        path = Path(base_path, subdir, f"{self.save_name}.pt")
        _state_dict = torch.load(path, map_location=device)

        if force_ordering:
            state_dict = OrderedDict()
            for new_key, value in zip(self.state_dict().keys(), _state_dict.values()):
                state_dict[new_key] = value
        else:
            state_dict = _state_dict

        self.load_state_dict(state_dict, assign=True)
        self.eval()


class KLCPERejector(Rejector):
    def __init__(
        self,
        clf: CPEClassifier,
        cpe_loss: CPELoss,
        bayes_posterior: None | CPEClassifier = None,
        lamb: float = 1,
    ):
        super().__init__()

        self.clf = clf
        self.bayes_posterior = (
            bayes_posterior if bayes_posterior is not None else clf
        )
        self.cpe_loss = cpe_loss
        self.lamb = lamb

        self.div = KLDiv()

        self.norm_z = nn.Parameter(torch.tensor(1), requires_grad=False)
        self.normalized = False

    def __str__(self):
        return "KL Rejector"

    def fit(self, train_dataloader: torch.utils.data.DataLoader, kwargs: dict[Any, Any] = {}) -> None:
        device = kwargs["device"]

        rej_sum = torch.tensor(0.0)
        count = 0

        for x, _ in train_dataloader:
            x = x.to(device)

            rej_sum += self.unnorm_rejection_value(x).sum().detach().cpu()
            count += x.shape[0]

        self.norm_z.data = (rej_sum / count).to(device)
        self.normalized = True

    def unnorm_rejection_value(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bayes_posterior.probit(x)
        y_hat = self.clf.probit(x)
        return self.div.f_prime_inv(-self.cpe_loss(y, y_hat) / self.lamb)

    def rejection_value(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalized:
            raise RejectorNotFitted

        return self.unnorm_rejection_value(x) / self.norm_z

    def rejection(self, x: torch.Tensor, tau: float):
        return self.rejection_value(x) < tau

    @property
    def save_name(self) -> str:
        return "kl_rejector" if self.lamb == 1 else f"kl_rejector_lambd{self.lamb:.3f}"


class AlphaBigCPERejector(Rejector):
    def __init__(
        self,
        clf: CPEClassifier,
        cpe_loss: CPELoss,
        alpha: float,
        bayes_posterior: None | CPEClassifier = None,
        ubound_multiply: float = 50,
        lamb: float = 1,
    ):
        if alpha < 1:
            raise RuntimeError("input alpha < 1")

        super().__init__()

        self.clf = clf
        self.bayes_posterior = (
            bayes_posterior if bayes_posterior is not None else clf
        )
        self.cpe_loss = cpe_loss
        self.alpha = alpha
        self.lamb = lamb

        self.div = AlphaDiv(alpha)

        self.b = nn.Parameter(torch.tensor(1), requires_grad=False)
        self.normalized = False
        self.ubound_multiply = ubound_multiply

    def fit(self, train_dataloader: torch.utils.data.DataLoader, kwargs: dict[Any, Any] = {}) -> None:
        device = kwargs["device"]
        loss_list = []

        for x, _ in train_dataloader:
            x = x.to(device)
            y = self.bayes_posterior.probit(x)
            y_hat = self.clf.probit(x)

            loss = self.cpe_loss(y, y_hat).detach().cpu()
            loss_list.append(loss)

        losses = torch.cat(loss_list)

        pv = 2 / (self.alpha - 1)

        if float(int(pv)) == pv and int(pv) % 2 == 0:
            def norm_root_func(b: float) -> float:
                inner = ((self.alpha - 1) / 2 * (b - losses / self.lamb))

                pw_rej = F.relu(
                        inner ** pv
                        )

                return float(pw_rej.mean()) - 1
        else:
            def norm_root_func(b: float) -> float:
                inner = ((self.alpha - 1) / 2 * (b - losses / self.lamb))

                pw_rej = F.relu(
                        inner
                        ) ** pv

                return float(pw_rej.mean()) - 1

        lb = float(losses.min() / self.lamb) + 1e-5
        ub = float(losses.max() * self.ubound_multiply / self.lamb)

        print(
                self.alpha,
                pv,
        norm_root_func(lb),
        norm_root_func(ub)
        )

        b, res = bisect(
            norm_root_func, lb, ub, maxiter=100_000, full_output=True
        )

        logging.debug(
            f"{self} fitting loss stats: min={losses.min():.2f},"
            f" mean={losses.mean():.2f}, max={losses.max():.2f}"
        )
        logging.debug(f"{self} fitting bisect results:\n{res}")
        self.b.data = torch.tensor(b).to(device)
        self.normalized = True

    def rejection_value(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalized:
            raise RejectorNotFitted

        y = self.bayes_posterior.probit(x)
        y_hat = self.clf.probit(x)
        loss = self.cpe_loss(y, y_hat)

        # Incorrect
        # return (2 / (self.alpha - 1) * F.relu(self.b - loss / self.lamb)) ** (
        #     2 / (self.alpha - 1)
        # )

        pv = 2 / (self.alpha - 1)

        if float(int(pv)) == pv and int(pv) % 2 == 0:
            inner = ((self.alpha - 1) / 2 * (self.b - loss / self.lamb))

            return F.relu(inner ** pv)
        else:
            inner = ((self.alpha - 1) / 2 * (self.b - loss / self.lamb))

            return F.relu(inner) ** pv


    def rejection(self, x: torch.Tensor, tau: float):
        return self.rejection_value(x) < tau

    def __str__(self):
        return f"(alpha={self.alpha:.3f}) Rejector"

    @property
    def save_name(self) -> str:
        return f"alpha_{self.alpha:.3f}_rejector" if self.lamb == 1 else f"alpha_{self.alpha:.3f}_rejector_lambd{self.lamb:.3f}"


class AlwaysAccept(Rejector):
    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor, tau: float):
        return self.rejection(x, tau)

    def rejection(self, x: torch.Tensor, tau: float):
        return torch.zeros(x.shape[0], dtype=torch.bool)

    def fit(self, train_dataloader: torch.utils.data.DataLoader, kwargs: dict[Any, Any] = {}) -> None:
        pass

    def __str__(self):
        return "Never Rejector"

    @property
    def save_name(self) -> str:
        return "never_rejector"
