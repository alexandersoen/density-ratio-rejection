from abc import ABC, abstractmethod

import torch


class FDiv(ABC):
    def __callable__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.div(x, y)

    def div(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.dot(x, self.f(y / x))

    @abstractmethod
    def f(self, z: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def f_prime(self, z: torch.Tensor) -> torch.Tensor:
        # Do we want to do this? Could autograd this instead of explicitly defining
        pass

    @abstractmethod
    def f_prime_inv(self, z: torch.Tensor) -> torch.Tensor:
        # Do we want to do this? Could autograd + inv
        pass


class KLDiv(FDiv):
    def f(self, z: torch.Tensor) -> torch.Tensor:
        return z * torch.log(z) - (z - 1)

    def f_prime(self, z: torch.Tensor) -> torch.Tensor:
        return torch.log(z)

    def f_prime_inv(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(z)


class AlphaDiv(FDiv):
    def __init__(self, alpha: float) -> None:
        if alpha == 1:
            raise RuntimeError("alpha = 1, should use KL instead")

        self.alpha = alpha

    def f(self, z: torch.Tensor) -> torch.Tensor:
        if self.alpha == 1:
            return -torch.log(z) + (z - 1)

        t1 = 4 / (1 - self.alpha) * (1 - torch.pow(z, (1 + self.alpha) / 2))
        t2 = (1 - self.alpha) / 2 * (z - 1)
        return t1 - t2

    def f_prime(self, z: torch.Tensor) -> torch.Tensor:
        return torch.pow(z, (1 - self.alpha) / 2)

    def f_prime_inv(self, z: torch.Tensor) -> torch.Tensor:
        return torch.pow(z, 2 / (1 - self.alpha))
