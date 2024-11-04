import argparse
import logging
import pathlib

import numpy as np
import torch
from torchinfo import summary

import rej.classifier as classifier
import rej.loss as L
import rej.rejector as rejector
import rej.utils as utils
from rej.augment import add_label_noise
from rej.data import (load_cifar10, load_gas_drift, load_har, load_mnist,
                      load_octmnist, load_organmnist)

DATASETS_GETTER = {
    "gas_drift": load_gas_drift,
    "har": load_har,
    "mnist": load_mnist,
    "organmnist": load_organmnist,
    "octmnist": load_octmnist,
    "cifar10": load_cifar10,
    "gas_drift_large": load_gas_drift,
    "har_large": load_har,
    "mnist_large": load_mnist,
}


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rejection via Distributions")

    parser.add_argument(
        "--dataset",
        type=str,
        default="gas_drift",
        metavar="D",
        help='dataset type (default: "gas_drift")',
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="number of cross validation folds (default: 5)",
    )
    parser.add_argument(
        "--fold-idx",
        type=int,
        required=True,
        help="folder idx",
    )

    parser.add_argument(
        "--val-split-percent",
        type=float,
        default=0.8,
        help="train-val split percentage (default: 0.8)",
    )

    parser.add_argument(
        "--num-tau-increments",
        type=int,
        default=20,
        help="number of tau increments when evaluating rejector (default: 20)",
    )

    parser.add_argument(
        "--model-folder",
        type=str,
        default="res",
        help='file name of the saved trained model (default: "res")',
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help='data folder (default: "data")',
    )

    parser.add_argument(
        "--noise-rate",
        type=float,
        default="0.25",
        help="noise rate for training set (default: 0.25)",
    )
    parser.add_argument(
        "--alpha-rej",
        type=float,
        action="extend",
        nargs="*",
        help="list of alpha-rejectors to train",
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default="1.",
        help="lambda temperature (default: 1.0)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="log",
        help="logging directory",
    )

    parser.add_argument(
        "--no-test-noise",
        action="store_true",
        help=(
            "flag to remove test noise to test set (does not effect training)"
        ),
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="if we want to report per iter tau",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="disable logging in stdder",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    logging_file = pathlib.Path(
        f"{args.log_dir}/{args.dataset}_{args.noise_rate}/{args.fold_idx}/rej.log"
    )
    logging_file.parent.mkdir(parents=True, exist_ok=True)

    handlers = [logging.FileHandler(logging_file)]
    if not args.quiet:
        # Not sure why appending gives an LSP error
        handlers += [logging.StreamHandler()]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.DEBUG,
        handlers=handlers,
        datefmt="%Y/%m/%d %I:%M:%S %p",
    )

    # Save sub-folder
    save_sub = pathlib.Path(
        f"{args.dataset}_{args.noise_rate}", f"fold_{args.fold_idx}"
    )
    logging.debug(
        f"training on dataset {args.dataset} (fold"
        f" {args.fold_idx}/{args.num_folds}) with {args.noise_rate} noise rate"
    )

    # Random seed
    logging.debug(f"using random seed {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Torch settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Dataset settings
    train_kwargs = {"batch_size": args.batch_size, "shuffle": True}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Loading
    data_path = pathlib.Path(args.data_path, args.dataset)
    train_dataset, test_dataset = DATASETS_GETTER[args.dataset](
        data_path, args.fold_idx, args.num_folds, random_state=args.seed
    )

    # Make training dataset noisy
    train_dataset = add_label_noise(train_dataset, noise_rate=args.noise_rate)
    if not args.no_test_noise:
        test_dataset = add_label_noise(
            test_dataset, noise_rate=args.noise_rate
        )

    # Train / Validation subsample
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.val_split_percent * num_train))

    np.random.seed(args.seed)
    np.random.shuffle(indices)

    train_idx, val_idx = indices[:split], indices[split:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    # DataLoaders
    sample_kwargs = train_kwargs
    sample_kwargs["shuffle"] = False
    clf_train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, **sample_kwargs
    )
    clf_val_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=val_sampler, **sample_kwargs
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # Define rej dataloader as joint on both train and val
    rej_loader = torch.utils.data.DataLoader(
        train_dataset,
        **train_kwargs,
    )

    base_kwargs = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "device": device,
        "log_interval": args.log_interval,
    }

    clf = classifier.CPEMulticlassClassifier(args.dataset).to(device)
    clf.load(args.model_folder, save_sub)

    for test_el, _ in rej_loader:
        logging.info(summary(clf, test_el.shape))
        break

    acceptor = rejector.AlwaysAccept()
    utils.report_performance_cpe(clf, acceptor, 0, device, test_loader)

    # Apply rejector
    rej = rejector.KLCPERejector(clf, L.LogLoss(), lamb=args.lamb).to(device)
    rej.fit(rej_loader, base_kwargs)

    if args.report:
        for i in range(args.num_tau_increments + 1):
            utils.report_performance_cpe(
                clf, rej, i / args.num_tau_increments, device, test_loader
            )

    # Save models
    rej.save(args.model_folder, save_sub)

    # Alpha rejectors
    alphas = [] if not args.alpha_rej else args.alpha_rej
    for alpha in alphas:
        alpha_rej = rejector.AlphaBigCPERejector(
            clf, L.LogLoss(), alpha, lamb=args.lamb
        ).to(device)
        alpha_rej.fit(rej_loader, base_kwargs)

        if args.report:
            for i in range(args.num_tau_increments + 1):
                utils.report_performance_cpe(
                    clf,
                    alpha_rej,
                    i / args.num_tau_increments,
                    device,
                    test_loader,
                )

        alpha_rej.save(args.model_folder, save_sub)
