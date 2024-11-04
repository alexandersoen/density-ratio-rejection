import pathlib

import numpy as np
import pandas as pd
import torch
from medmnist import OCTMNIST, OrganSMNIST
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

DEFAULT_SPLIT_RATE = 0.8


class UnknownDataset(Exception):
    pass


class OutsideFoldIndexRange(Exception):
    pass


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        input_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if type(data) == torch.Tensor:
            self.data = data.to(input_dtype)
        else:
            self.data = torch.tensor(data, dtype=input_dtype)

        if type(targets) != torch.Tensor:
            self.targets = torch.tensor(targets)
        else:
            self.targets = targets

        self.targets = self.targets.reshape(-1).long()
        self.classes = sorted(set(self.targets.tolist()))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def load_gas_drift(
    data_path: pathlib.Path,
    fold_idx: int,
    num_folds: int = 5,
    random_state: int | None = None,
):
    data_path = pathlib.Path(str(data_path).split("_large")[0])

    if not (1 <= fold_idx <= num_folds):
        raise OutsideFoldIndexRange

    # Process data
    data_dicts = []
    for f_path in data_path.glob("*.dat"):
        with f_path.open() as f:
            for row in f:
                cur_data = row.strip().split(" ")
                cur_data[0] = f"target:{cur_data[0]}"
                cur_data = [s.split(":", maxsplit=1) for s in cur_data]

                cur_data_dict = {str(k): float(v) for k, v in cur_data}
                data_dicts.append(cur_data_dict)

    data_pd = pd.DataFrame.from_records(data_dicts)
    data_pd["target"] = data_pd["target"].astype("int").astype("category")

    features = [c for c in data_pd.columns if c != "target"]
    xs = data_pd[features].to_numpy()
    ys = data_pd[["target"]].to_numpy() - 1  # Remap first class to 0

    batch_index = np.arange(xs.shape[0])

    cv = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)

    train_idx, test_idx = batch_index, batch_index
    splits = cv.split(batch_index)
    for _ in range(fold_idx):
        train_idx, test_idx = next(splits)

    train_xs, train_ys = xs[train_idx], ys[train_idx]
    test_xs, test_ys = xs[test_idx], ys[test_idx]

    x_transform = Pipeline(
        [
            ("z_transform", StandardScaler()),
        ]
    )

    train_xs = x_transform.fit_transform(train_xs)

    test_xs = x_transform.transform(test_xs)

    train_dataset = ClassificationDataset(train_xs, train_ys)
    test_dataset = ClassificationDataset(test_xs, test_ys)

    return train_dataset, test_dataset


def load_har(
    data_path: pathlib.Path,
    fold_idx: int,
    num_folds: int = 5,
    random_state: int | None = None,
):
    data_path = pathlib.Path(str(data_path).split("_large")[0])

    def load_data(subset_name: str) -> tuple[np.ndarray, np.ndarray]:
        xs_list = []
        xs_path = pathlib.Path(data_path, f"{subset_name}/X_{subset_name}.txt")
        with xs_path.open() as f:
            for row in f:
                cur_vals = [
                    float(v) for v in row.strip().split(" ") if v != ""
                ]
                xs_list.append(cur_vals)

        ys_list = []
        ys_path = pathlib.Path(data_path, f"{subset_name}/y_{subset_name}.txt")
        with ys_path.open() as f:
            for row in f:
                y_val = int(row.strip())
                ys_list.append(y_val)

        xs = np.array(xs_list)
        # Remapping the ys to it index starts from 0
        ys = np.array(ys_list).reshape(-1, 1) - 1

        return xs, ys

    _train_xs, _train_ys = load_data("train")
    _test_xs, _test_ys = load_data("test")

    xs = np.concatenate((_train_xs, _test_xs), axis=0)
    ys = np.concatenate((_train_ys, _test_ys), axis=0)

    batch_index = np.arange(xs.shape[0])

    cv = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)

    train_idx, test_idx = batch_index, batch_index
    splits = cv.split(batch_index)
    for _ in range(fold_idx):
        train_idx, test_idx = next(splits)

    train_xs, train_ys = xs[train_idx], ys[train_idx]
    test_xs, test_ys = xs[test_idx], ys[test_idx]

    x_transform = Pipeline(
        [
            ("z_transform", StandardScaler()),
        ]
    )

    train_xs = x_transform.fit_transform(train_xs)

    test_xs = x_transform.transform(test_xs)

    train_dataset = ClassificationDataset(train_xs, train_ys)
    test_dataset = ClassificationDataset(test_xs, test_ys)

    return train_dataset, test_dataset


def load_mnist(
    data_path: pathlib.Path,
    fold_idx: int,
    num_folds: int = 5,
    random_state: int | None = None,
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset_train_part = datasets.MNIST(
        str(data_path), train=True, download=True
    )
    dataset_test_part = datasets.MNIST(str(data_path), train=False)

    xs = torch.concat([dataset_train_part.data, dataset_test_part.data])
    ys = torch.concat([dataset_train_part.targets, dataset_test_part.targets])

    cv = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)

    batch_index = torch.arange(xs.shape[0])
    train_idx, test_idx = batch_index, batch_index
    splits = cv.split(batch_index)
    for _ in range(fold_idx):
        train_idx, test_idx = next(splits)

    train_idx, test_idx = torch.tensor(train_idx), torch.tensor(test_idx)

    train_dataset = datasets.MNIST(
        str(data_path), train=True, transform=transform
    )
    train_dataset.data = xs[train_idx]
    train_dataset.targets = ys[train_idx]

    test_dataset = datasets.MNIST(
        str(data_path), train=False, transform=transform
    )
    test_dataset.data = xs[test_idx]
    test_dataset.targets = ys[test_idx]

    return train_dataset, test_dataset


def load_organmnist(
    data_path: pathlib.Path,
    fold_idx: int,
    num_folds: int = 5,
    random_state: int | None = None,
):

    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    parts = ["train", "val", "test"]

    data_path.mkdir(parents=True, exist_ok=True)
    ds = [
        OrganSMNIST(root=str(data_path), split=p, download=True) for p in parts
    ]

    xs = torch.concat([torch.tensor(d.imgs) for d in ds])
    ys = torch.concat([torch.tensor(d.labels) for d in ds]).squeeze()

    cv = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)

    batch_index = torch.arange(xs.shape[0])
    train_idx, test_idx = batch_index, batch_index
    splits = cv.split(batch_index)
    for _ in range(fold_idx):
        train_idx, test_idx = next(splits)
    classes = sorted(torch.unique(torch.tensor(ys, dtype=torch.int64)))

    train_idx, test_idx = torch.tensor(train_idx), torch.tensor(test_idx)

    train_dataset = OrganSMNIST(
        root=str(data_path),
        split="train",
        transform=transform_train,
    )
    train_dataset.imgs = xs[train_idx].numpy()
    train_dataset.labels = ys[train_idx].numpy()
    setattr(train_dataset, "data", torch.tensor(train_dataset.imgs))
    setattr(
        train_dataset,
        "targets",
        torch.tensor(train_dataset.labels, dtype=torch.int64),
    )
    setattr(train_dataset, "classes", classes)

    test_dataset = OrganSMNIST(
        root=str(data_path),
        split="test",
        transform=transform_test,
    )
    test_dataset.imgs = xs[test_idx].numpy()
    test_dataset.labels = ys[test_idx].numpy()
    setattr(test_dataset, "data", torch.tensor(test_dataset.imgs))
    setattr(
        test_dataset,
        "targets",
        torch.tensor(test_dataset.labels, dtype=torch.int64),
    )
    setattr(test_dataset, "classes", classes)

    # Update info (there is an assert on __len__)
    train_dataset.info["n_samples"]["train"] = len(train_dataset.imgs)
    train_dataset.info["n_samples"]["test"] = len(test_dataset.imgs)
    train_dataset.info["n_samples"]["val"] = 0

    test_dataset.info["n_samples"]["train"] = len(train_dataset.imgs)
    test_dataset.info["n_samples"]["test"] = len(test_dataset.imgs)
    test_dataset.info["n_samples"]["val"] = 0

    return train_dataset, test_dataset


def load_octmnist(
    data_path: pathlib.Path,
    fold_idx: int,
    num_folds: int = 5,
    random_state: int | None = None,
):

    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    parts = ["train", "val", "test"]

    data_path.mkdir(parents=True, exist_ok=True)
    ds = [OCTMNIST(root=str(data_path), split=p, download=True) for p in parts]

    xs = torch.concat([torch.tensor(d.imgs) for d in ds])
    ys = torch.concat([torch.tensor(d.labels) for d in ds]).squeeze()

    cv = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)

    batch_index = torch.arange(xs.shape[0])
    train_idx, test_idx = batch_index, batch_index
    splits = cv.split(batch_index)
    for _ in range(fold_idx):
        train_idx, test_idx = next(splits)
    classes = sorted(torch.unique(torch.tensor(ys, dtype=torch.int64)))

    train_idx, test_idx = torch.tensor(train_idx), torch.tensor(test_idx)

    train_dataset = OCTMNIST(
        root=str(data_path),
        split="train",
        transform=transform_train,
    )
    train_dataset.imgs = xs[train_idx].numpy()
    train_dataset.labels = ys[train_idx].numpy()
    setattr(train_dataset, "data", torch.tensor(train_dataset.imgs))
    setattr(
        train_dataset,
        "targets",
        torch.tensor(train_dataset.labels, dtype=torch.int64),
    )
    setattr(train_dataset, "classes", classes)

    test_dataset = OCTMNIST(
        root=str(data_path),
        split="test",
        transform=transform_test,
    )
    test_dataset.imgs = xs[test_idx].numpy()
    test_dataset.labels = ys[test_idx].numpy()
    setattr(test_dataset, "data", torch.tensor(test_dataset.imgs))
    setattr(
        test_dataset,
        "targets",
        torch.tensor(test_dataset.labels, dtype=torch.int64),
    )
    setattr(test_dataset, "classes", classes)

    # Update info (there is an assert on __len__)
    train_dataset.info["n_samples"]["train"] = len(train_dataset.imgs)
    train_dataset.info["n_samples"]["test"] = len(test_dataset.imgs)
    train_dataset.info["n_samples"]["val"] = 0

    test_dataset.info["n_samples"]["train"] = len(train_dataset.imgs)
    test_dataset.info["n_samples"]["test"] = len(test_dataset.imgs)
    test_dataset.info["n_samples"]["val"] = 0

    return train_dataset, test_dataset


def load_cifar10(
    data_path: pathlib.Path,
    fold_idx: int,
    num_folds: int = 5,
    random_state: int | None = None,
):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    dataset_train_part = datasets.CIFAR10(
        str(data_path), train=True, download=True
    )
    dataset_test_part = datasets.CIFAR10(str(data_path), train=False)

    xs = np.concatenate([dataset_train_part.data, dataset_test_part.data])
    ys = np.concatenate(
        [
            np.array(dataset_train_part.targets),
            np.array(dataset_test_part.targets),
        ]
    )

    cv = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)

    batch_index = torch.arange(xs.shape[0])
    train_idx, test_idx = batch_index, batch_index
    splits = cv.split(batch_index)
    for _ in range(fold_idx):
        train_idx, test_idx = next(splits)

    train_dataset = datasets.CIFAR10(
        str(data_path), train=True, transform=transform_train
    )
    train_dataset.data = xs[train_idx]
    train_dataset.targets = list(ys[train_idx])

    test_dataset = datasets.CIFAR10(
        str(data_path), train=False, transform=transform_test
    )
    test_dataset.data = xs[test_idx]
    test_dataset.targets = list(ys[test_idx])

    return train_dataset, test_dataset
