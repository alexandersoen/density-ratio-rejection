import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import rej.classifier as classifier
import rej.loss as L
import rej.rejector as rejector


@dataclass
class SummarizePerformance:
    predicted: np.ndarray
    labels: np.ndarray
    losses: np.ndarray
    zero_one: np.ndarray
    to_reject: np.ndarray

    def accepted_predictions(self):
        return self.predicted[~self.to_reject]

    def accepted_losses(self):
        return self.losses[~self.to_reject]

    def accepted_zero_one(self):
        return self.zero_one[~self.to_reject].astype(float)

    def accepted_zero_one_risk(self):
        zero_one = self.accepted_zero_one()
        return zero_one.mean() if len(zero_one) > 0 else 1

    def coverage_percentage(self):
        return 100 * (~self.to_reject).astype(int).mean()

    def accepted_p_losses(self, p):
        losses = self.accepted_losses()
        return np.percentile(losses, p) if len(losses) > 0 else 0

    def accepted_per_class_risk(self):
        accepted_labels = self.labels[~self.to_reject]
        label_set = list(np.unique(self.labels))

        zero_one = self.accepted_zero_one()
        risk_array = np.zeros(len(label_set))

        for i, l in enumerate(label_set):
            cur_zo = zero_one[accepted_labels == l]
            v = cur_zo.mean() if len(cur_zo) > 0 else 1

            risk_array[i] = v

        return risk_array


def summarize_performance_cpe(
    model: classifier.CPEClassifier,
    rej: rejector.Rejector,
    tau: float,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
) -> SummarizePerformance:
    model.eval()
    all_labels = []
    all_losses = []
    all_preds = []
    all_rejects = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            log_pred = model(data)
            to_reject = rej(data, tau)

            pred_class = log_pred.argmax(dim=1, keepdim=True)
            target_class = target.reshape(-1, 1)

            target_one_hot = F.one_hot(target, model.n_classes)
            cur_loss = model.calculate_loss(data, target_one_hot).detach()

            all_labels.append(target_class)
            all_losses.append(cur_loss)
            all_preds.append(pred_class)
            all_rejects.append(to_reject)

        labels = torch.cat(all_labels)
        losses = torch.cat(all_losses)
        preds = torch.cat(all_preds)
        rejects = torch.cat(all_rejects)

    # Report summary statistics
    pw_correct = preds.eq(labels)

    summary = SummarizePerformance(
        np.array(preds.cpu().detach()),
        np.array(labels.cpu().detach()),
        np.array(losses.cpu().detach()),
        np.array(pw_correct.cpu().detach()),
        np.array(rejects.cpu().detach(), dtype=bool),
    )

    return summary


def report_performance_cpe(
    model: classifier.CPEClassifier,
    rej: rejector.Rejector,
    tau: float,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    param_str: str = "tau",
) -> SummarizePerformance:
    model.eval()
    all_logits = []
    all_labels = []
    all_losses = []
    all_preds = []
    all_rejects = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            cur_logits = model.logit(data)
            log_pred = model(data)
            to_reject = rej(data, tau)

            target_one_hot = F.one_hot(target, model.n_classes)

            cur_loss = model.calculate_loss(data, target_one_hot).detach()
            pred_class = log_pred.argmax(dim=1, keepdim=True)
            target_class = target.reshape(-1, 1)

            all_logits.append(cur_logits)
            all_labels.append(target_class)
            all_losses.append(cur_loss)
            all_preds.append(pred_class)
            all_rejects.append(to_reject)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        losses = torch.cat(all_losses)
        preds = torch.cat(all_preds)
        rejects = torch.cat(all_rejects)

    # Report summary statistics
    pw_correct = preds.eq(labels)
    pw_loss = losses

    total_correct = pw_correct.sum()
    avg_loss = pw_loss.mean()

    accept_total_correct = pw_correct[~rejects].sum()
    accept_avg_loss = pw_loss[~rejects].mean()

    accepts = (~rejects).to(torch.int).sum()
    total_samples = rejects.shape[0]

    ece = L.ECELoss()
    ece_score = ece(logits, labels).item()
    accept_ece_score = ece(logits[~rejects], labels[~rejects]).item()

    logging.info(
        "\n Performance Report of {} with {}={}:\n"
        "Average Loss: {:.4f} -> {:.4f}\n"
        "Accuracy: {}/{} ({:.2f}%) -> {}/{} ({:.2f}%)\n"
        "ECE: {:.4f} -> {:.4f}\n"
        "Accept Coverage: {}/{} ({:.2f}%)\n"
        "\n".format(
            rej,
            param_str,
            tau,
            avg_loss,
            accept_avg_loss,
            total_correct,
            total_samples,
            100 * total_correct / total_samples,
            accept_total_correct,
            accepts,
            100 * accept_total_correct / accepts,
            ece_score,
            accept_ece_score,
            accepts,
            total_samples,
            100 * accepts / total_samples,
        )
    )

    summary = SummarizePerformance(
        np.array(preds.cpu().detach()),
        np.array(labels.cpu().detach()),
        np.array(losses.cpu().detach()),
        np.array(pw_correct.cpu().detach()),
        np.array(rejects.cpu().detach(), dtype=bool),
    )

    return summary
