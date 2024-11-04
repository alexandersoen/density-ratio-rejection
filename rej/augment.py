import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def add_label_noise(dataset, noise_rate: float = 0.25):
    # Utilizing uniform label noise
    assert 0 <= noise_rate <= 1

    n_classes = len(dataset.classes)
    clean_ys = F.one_hot(torch.tensor(dataset.targets), n_classes).to(
        torch.double
    )

    probs_tensor = torch.zeros_like(clean_ys)
    probs_tensor += clean_ys * (1 - noise_rate)
    probs_tensor += (1 - clean_ys) * (noise_rate / n_classes)

    noisy_ys = Categorical(probs_tensor).sample()

    dataset.targets = noisy_ys

    return dataset
