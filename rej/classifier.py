import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from rej.data import UnknownDataset
from rej.loss import CPELoss, LogLoss
from rej.resnet import ResNet34
import rej.resnet as resnet_cifar
from rej.resnet2 import ResNet18


def gas_drift_base() -> nn.Sequential:
    GAS_INPUT_DIM = 128
    GAS_OUTPUT_DIM = 6

    model = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(GAS_INPUT_DIM, 64)),
                ("batch_norm", nn.BatchNorm1d(64)),
                ("act1", nn.Sigmoid()),
                ("fc2", nn.Linear(64, GAS_OUTPUT_DIM)),
            ]
        )
    )

    return model


def har_base() -> nn.Sequential:
    HAR_INPUT_DIM = 561
    HAR_OUTPUT_DIM = 6

    model = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(HAR_INPUT_DIM, 64)),
                ("batch_norm", nn.BatchNorm1d(64)),
                ("act1", nn.Sigmoid()),
                ("fc2", nn.Linear(64, HAR_OUTPUT_DIM)),
            ]
        )
    )

    return model


def mnist_base() -> nn.Sequential:
    make_act = lambda: nn.Sigmoid()
    model = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 32, 3, 1)),
                ("act1", make_act()),
                ("conv2", nn.Conv2d(32, 64, 3, 1)),
                ("act2", make_act()),
                ("maxpool", nn.MaxPool2d(2)),
                ("dropout1", nn.Dropout(0.25)),
                ("flatten", nn.Flatten()),
                ("fc1", nn.Linear(9216, 128)),
                ("act3", make_act()),
                ("dropout2", nn.Dropout(0.5)),
                ("fc2", nn.Linear(128, 10)),
            ]
        )
    )
    return model


def gas_drift_base_large() -> nn.Sequential:
    GAS_INPUT_DIM = 128
    GAS_OUTPUT_DIM = 6

    model = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(GAS_INPUT_DIM, 64)),
                ("act1", nn.Sigmoid()),
                ("fc2", nn.Linear(64, 64)),
                ("batch_norm", nn.BatchNorm1d(64)),
                ("act2", nn.Sigmoid()),
                ("fc3", nn.Linear(64, GAS_OUTPUT_DIM)),
            ]
        )
    )

    return model


def har_base_large() -> nn.Sequential:
    HAR_INPUT_DIM = 561
    HAR_OUTPUT_DIM = 6

    model = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(HAR_INPUT_DIM, 64)),
                ("act1", nn.Sigmoid()),
                ("fc2", nn.Linear(64, 64)),
                ("batch_norm", nn.BatchNorm1d(64)),
                ("act2", nn.Sigmoid()),
                ("fc3", nn.Linear(64, HAR_OUTPUT_DIM)),
            ]
        )
    )

    return model


def mnist_base_large() -> nn.Sequential:
    make_act = lambda: nn.Sigmoid()
    model = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 32, 3, 1)),
                ("act1", make_act()),
                ("conv2", nn.Conv2d(32, 64, 3, 1)),
                ("act2", make_act()),
                ("maxpool", nn.MaxPool2d(2)),
                ("dropout1", nn.Dropout(0.25)),
                ("flatten", nn.Flatten()),
                ("fc1", nn.Linear(9216, 128)),
                ("act3", make_act()),
                ("fc2", nn.Linear(128, 128)),
                ("act4", make_act()),
                ("dropout2", nn.Dropout(0.5)),
                ("fc3", nn.Linear(128, 10)),
            ]
        )
    )
    return model


def cifar10_base() -> nn.Sequential:
    # return ResNet34().return_model()
    return resnet_cifar.ResNet18().return_model()


BASE_MODELS = {
    "gas_drift": gas_drift_base,
    "har": har_base,
    "mnist": mnist_base,
    "organmnist": lambda: ResNet18(1, 11).model,
    "octmnist": lambda: ResNet18(1, 4).model,
    "cifar10": cifar10_base,
    "gas_drift_large": gas_drift_base_large,
    "har_large": har_base_large,
    "mnist_large": mnist_base_large,
}


class CPEClassifier(ABC, nn.Module):
    loss: CPELoss
    n_classes: int

    @abstractmethod
    def calculate_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def probit(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def save_name(self) -> str:
        pass

    def save_path(self, base_path: Path | str, subdir: Path | str) -> Path:
        path = Path(base_path, subdir, f"{self.save_name}.pt")
        return path

    def save(self, base_path: Path | str, subdir: Path | str) -> None:
        path = Path(base_path, subdir, f"{self.save_name}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(
        self,
        base_path: Path | str,
        subdir: Path | str,
        device: torch.device = torch.device("cpu"),
        force_ordering=False,
    ) -> None:
        path = Path(base_path, subdir, f"{self.save_name}.pt")
        _state_dict = torch.load(path, map_location=device)

        if force_ordering:
            state_dict = OrderedDict()
            for new_key, value in zip(
                self.state_dict().keys(), _state_dict.values()
            ):
                state_dict[new_key] = value
        else:
            state_dict = _state_dict

        self.load_state_dict(state_dict, assign=True)
        self.eval()


class CPEMulticlassClassifier(CPEClassifier):
    def __init__(self, dataset: str) -> None:
        super().__init__()

        self.dataset = dataset

        try:
            self.model = BASE_MODELS[dataset]()
            self.n_classes = self.model[-1].out_features
        except KeyError:
            raise UnknownDataset(f'Unknown dataset name given: "{dataset}"')

        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

        self.loss: LogLoss = LogLoss()

    def unscaled_logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.temperature_scale(self.unscaled_logit(x))

    def temperature_scale(self, logit: torch.Tensor) -> torch.Tensor:
        return logit / torch.abs(self.temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.logit(x)
        log_y_hat = F.log_softmax(z, dim=1)

        return log_y_hat

    def probit(self, x: torch.Tensor) -> torch.Tensor:
        z = self.logit(x)
        y_hat = F.softmax(z, dim=1)

        return y_hat

    def calculate_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.logit(x)
        loss = self.loss.pw_logit_risk(y, logits)
        return loss

    def train_network(self) -> None:
        self.train()
        self.unfreeze()
        self.temperature.requires_grad = False

    def train_temperature(self) -> None:
        self.train()
        self.freeze()
        self.temperature.requires_grad = True

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    @property
    def save_name(self) -> str:
        return f"cpe_multiclass_classifier"


def train_cpe_network(
    model: CPEClassifier,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    log_interval: int = 10,
) -> None:
    model.train_network()
    pbar = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        target = F.one_hot(target, model.n_classes)
        loss = model.calculate_loss(data, target).mean()

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            logging.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * train_loader.batch_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def train_cpe_temperature(
    model: CPEClassifier,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    max_iter: int = 50,
    learning_rate: float = 1e-1,
) -> None:
    model.train_temperature()
    model.temperature.data = torch.ones(1) * 1.5
    model.to(device)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=learning_rate, max_iter=max_iter
    )

    all_logits = []
    all_targets = []
    with torch.no_grad():
        for data, target in tqdm(val_loader, leave=False):
            data, target = data.to(device), target.to(device)

            cur_logits = model.unscaled_logit(data)

            all_logits.append(cur_logits)
            all_targets.append(target)

        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)

    def eval() -> float:
        optimizer.zero_grad()
        total_loss = F.cross_entropy(
            model.temperature_scale(logits), targets, reduction="mean"
        )
        total_loss.backward()
        return float(total_loss)

    optimizer.step(eval)

    logging.info("Temperature set to: {:.4f}".format(model.temperature.item()))


def add_second_last_batch_norm(seq_module: nn.Sequential) -> nn.Sequential:
    """
    Add batch norm after second last Linear
    """
    mem = []
    size = len(seq_module)
    while True:
        # If linear and flatten before:
        if (
            len(seq_module) > 2
            and isinstance(seq_module[-1], nn.Linear)
            and isinstance(seq_module[-2], nn.Flatten)
        ):
            new_batch_norm = nn.BatchNorm1d(seq_module[-1].in_features)
            cur_l = seq_module.pop(-1)
            mem.append(cur_l)
            seq_module.append(new_batch_norm)
        # If linear, add a batch norm
        if size != len(seq_module) and (isinstance(seq_module[-1], nn.Linear)):
            new_batch_norm = nn.BatchNorm1d(seq_module[-1].out_features)
            seq_module.append(new_batch_norm)
            break
        # Short circuit if batch norm already exists
        elif isinstance(seq_module[-1], nn.BatchNorm1d):
            break

        try:
            cur_l = seq_module.pop(-1)
            mem.append(cur_l)
        except IndexError:
            break

    for l in reversed(mem):
        seq_module.append(l)

    return seq_module
