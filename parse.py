import argparse
import json
import pathlib

import numpy as np
import torch
from tqdm import trange

import rej.classifier as classifier
import rej.loss as L
import rej.rejector as rejector
from rej.augment import add_label_noise
from rej.baseline.css import CSSReject, CSSurrogateMulticlass
from rej.baseline.defer import Defer, DeferReject
from rej.baseline.gce import GCEConsistent, GCEConsistentReject
from rej.baseline.predrej import PredictReject
from rej.data import (load_cifar10, load_gas_drift, load_har, load_mnist,
                      load_octmnist, load_organmnist)
from rej.utils import SummarizePerformance, summarize_performance_cpe

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


def report(summary: SummarizePerformance):
    return (
        100 * summary.accepted_zero_one_risk(),
        summary.coverage_percentage(),
        summary.accepted_p_losses(0),
        summary.accepted_p_losses(25),
        summary.accepted_p_losses(50),
        summary.accepted_p_losses(75),
        summary.accepted_p_losses(100),
        summary.accepted_per_class_risk(),
    )


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show plots")

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
        default=4096,
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
        help="learning rate (default: 0.01)",
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
        "--num-folds",
        type=int,
        default=5,
        help="number of cross validation folds (default: 5)",
    )

    parser.add_argument(
        "--fold-idx",
        type=int,
        default=None,
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
        "--no-test-noise",
        action="store_true",
        help="flag to remove test noise to test set",
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
        action="extend",
        nargs="*",
        help="list of lamb to train",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="do not calculate over baselines",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="res.json",
        help='save path (default: "res.json")',
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    # Torch settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Dataset settings
    test_kwargs = {"batch_size": args.test_batch_size, "shuffle": False}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": False}
        test_kwargs.update(cuda_kwargs)

    # Plotting data
    plotting_data_list = []

    if args.fold_idx is not None:
        folds_to_process = [args.fold_idx]
    else:
        folds_to_process = trange(1, 1 + args.num_folds, desc="folds")

    for fold_idx in folds_to_process:
        plotting_data = {}

        # Save sub-folder
        save_sub = pathlib.Path(
            f"{args.dataset}_{args.noise_rate}", f"fold_{fold_idx}"
        )

        # Random seed
        print(f"using random seed {args.seed}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Loading
        data_path = pathlib.Path(args.data_path, args.dataset)
        _, test_dataset = DATASETS_GETTER[args.dataset](
            data_path, fold_idx, args.num_folds, random_state=args.seed
        )

        # Torch stuff
        if not args.no_test_noise:
            test_dataset = add_label_noise(
                test_dataset, noise_rate=args.noise_rate
            )

        # DataLoaders
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        # Base classifier
        clf = classifier.CPEMulticlassClassifier(args.dataset).to(device)
        clf.load(args.model_folder, save_sub, device=device)

        # Train Classifier
        acceptor = rejector.AlwaysAccept()
        base_clf_summary = summarize_performance_cpe(
            clf, acceptor, 0, device, test_loader
        )
        plotting_data["base_clf"] = {None: report(base_clf_summary)}

        # Load and test standard KL tester over tau range
        for lamb in args.lamb:
            plotting_data[f"kl_rej_lamb_{lamb}"] = {}
            kl_rej = rejector.KLCPERejector(clf, L.LogLoss(), lamb=lamb)

            try:
                kl_rej.load(args.model_folder, save_sub, device)
            except FileNotFoundError:
                continue

            kl_rej.normalized = True
            for t in trange(
                1, args.num_tau_increments, desc=f"kl_rej_lamb_{lamb}"
            ):
                tau = t / args.num_tau_increments

                kl_rej_summary = summarize_performance_cpe(
                    clf, kl_rej, tau, device, test_loader
                )
                plotting_data[f"kl_rej_lamb_{lamb}"][tau] = report(
                    kl_rej_summary
                )

            for alpha in args.alpha_rej:
                alpha_name = f"alpha_{alpha:.3f}_rej_lamb_{lamb}"

                alpha_rej = rejector.AlphaBigCPERejector(
                    clf, L.LogLoss(), alpha, lamb=lamb
                ).to(device)

                try:
                    alpha_rej.load(args.model_folder, save_sub, device)
                except FileNotFoundError:
                    continue

                alpha_rej.normalized = True

                plotting_data[alpha_name] = {}
                for t in trange(1, args.num_tau_increments, desc=alpha_name):
                    tau = t / args.num_tau_increments

                    alpha_rej_summary = summarize_performance_cpe(
                        clf, alpha_rej, tau, device, test_loader
                    )
                    plotting_data[alpha_name][tau] = report(alpha_rej_summary)

        if not args.skip_baselines:
            print("Processing baselines")

            # Load and test Prediction-Rejection baseline
            plotting_data["pred_rej"] = {}
            for i in trange(1, args.num_tau_increments, desc="pred_rej"):
                c = i / (2 * args.num_tau_increments)

                pred_rej = PredictReject(clf, c).to(device)

                try:
                    pred_rej.load(
                        args.model_folder,
                        save_sub,
                        device,
                        force_ordering=True,
                    )
                except:
                    continue

                pred_rej_summary = summarize_performance_cpe(
                    clf, pred_rej, c, device, test_loader
                )
                plotting_data["pred_rej"][c] = report(pred_rej_summary)

            plotting_data["css"] = {}
            for i in trange(1, args.num_tau_increments, desc="css"):
                c = i / (2 * args.num_tau_increments)
                css_model = CSSurrogateMulticlass(args.dataset, c).to(device)
                css_rejector = CSSReject(css_model)

                try:
                    css_model.load(
                        args.model_folder,
                        save_sub,
                        device,
                        force_ordering=True,
                    )
                    css_rejector.load(
                        args.model_folder,
                        save_sub,
                        device,
                        force_ordering=True,
                    )
                except:
                    continue

                css_summary = summarize_performance_cpe(
                    css_model, css_rejector, c, device, test_loader
                )

                plotting_data["css"][c] = report(css_summary)

            plotting_data["defer"] = {}
            for i in trange(1, args.num_tau_increments, desc="defer"):
                c = i / (2 * args.num_tau_increments)
                defer_model = Defer(args.dataset, c).to(device)
                defer_rejector = DeferReject(defer_model)

                try:
                    defer_model.load(
                        args.model_folder,
                        save_sub,
                        device,
                        force_ordering=True,
                    )
                    defer_rejector.load(
                        args.model_folder,
                        save_sub,
                        device,
                        force_ordering=True,
                    )

                    defer_summary = summarize_performance_cpe(
                        defer_model, defer_rejector, c, device, test_loader
                    )
                except:
                    continue

                plotting_data["defer"][c] = report(defer_summary)

            plotting_data["gce"] = {}
            for i in trange(1, args.num_tau_increments, desc="gce"):
                c = i / (2 * args.num_tau_increments)
                gce_model = GCEConsistent(args.dataset, c).to(device)
                gce_rejector = GCEConsistentReject(gce_model)

                try:
                    gce_model.load(
                        args.model_folder,
                        save_sub,
                        device,
                        force_ordering=True,
                    )
                    gce_rejector.load(
                        args.model_folder,
                        save_sub,
                        device,
                        force_ordering=True,
                    )
                except:
                    continue

                gce_summary = summarize_performance_cpe(
                    gce_model, gce_rejector, c, device, test_loader
                )

                plotting_data["gce"][c] = report(gce_summary)

        plotting_data_list.append(plotting_data)

    all_plotting_data = {}
    for p in plotting_data_list:
        for t, tres in p.items():
            if t not in all_plotting_data:
                all_plotting_data[t] = {}

            for c, cres in tres.items():
                if c not in all_plotting_data[t]:
                    all_plotting_data[t][c] = {
                        "accuracy": [],
                        "coverage": [],
                        "p0": [],
                        "p25": [],
                        "p50": [],
                        "p75": [],
                        "p100": [],
                        "risks": [],
                    }

                all_plotting_data[t][c]["accuracy"].append(cres[0])
                all_plotting_data[t][c]["coverage"].append(cres[1])
                all_plotting_data[t][c]["p0"].append(cres[2])
                all_plotting_data[t][c]["p25"].append(cres[3])
                all_plotting_data[t][c]["p50"].append(cres[4])
                all_plotting_data[t][c]["p75"].append(cres[5])
                all_plotting_data[t][c]["p100"].append(cres[6])
                all_plotting_data[t][c]["risks"].append(list(cres[7]))

    save_path = pathlib.Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w") as f:
        json.dump(all_plotting_data, f)
