import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Configure Python, NumPy and PyTorch random number generators
    for reproducible CartoGen experiments.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Return CUDA when available, otherwise CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def print_device_information() -> None:

    device = get_device()

    print("=" * 80)
    print("DEVICE INFORMATION")
    print("=" * 80)

    print(f"Selected device: {device}")

    if torch.cuda.is_available():

        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        properties = torch.cuda.get_device_properties(0)

        memory_gb = properties.total_memory / (1024 ** 3)

        print(f"GPU memory: {memory_gb:.2f} GB")

    else:

        print("CUDA GPU not detected.")