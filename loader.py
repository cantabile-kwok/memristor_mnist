import codecs
import gzip
import lzma
import math
import os
import os.path
import string
import warnings
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import URLError

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
    extract_archive,
    verify_str_arg,
)


def LoadMNIST(batch_size=32, validation=True, num_workers=1):
    """Method to load the MNIST dataset.

    Parameters
    ----------
    batch_size : int
        Batch size.
    validation : bool
        Load the validation set (True).
    num_workers : int
        Number of workers to use.

    Returns
    -------
    list of torch.utils.data
        The train, validiation, and test loaders.
    """
    root = "data"
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_set = torchvision.datasets.MNIST(
        root=root, train=True, transform=transform, download=True
    )
    test_set = torchvision.datasets.MNIST(
        root=root, train=False, transform=transform, download=True
    )
    if validation:
        train_size = int(0.8 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        train_set, validation_set = torch.utils.data.random_split(
            full_train_set, [train_size, validation_size]
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            full_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        validation_loader = None

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=int(batch_size / 2), shuffle=False, num_workers=num_workers
    )
    return train_loader, validation_loader, test_loader
