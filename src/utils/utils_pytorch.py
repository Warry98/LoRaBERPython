import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

def loramod(x: Tensor, SF: int, BW: float, fs:float, direction: int = 1):
    """
    Adapted From: https://github.com/bhomssi/LoRaBERMatlab/tree/main
    TODO: fix sensibility to scale of SF, beta and gamma (precision problems)
    """
    M = 2 ** SF

    if not torch.is_tensor(x) or x.dtype != torch.int64 or torch.any(x < 0) or torch.any(x >= M):
        raise ValueError("Input symbols must be integers in the range [0, M-1]")

    Ts = (2**int(SF)) / int(BW)
    beta = BW/(2*Ts)
    n_symbol = fs * M / BW
    arange = torch.arange(n_symbol, device = x.device)
    t_symbol = arange.unsqueeze(-2)  * 1/fs
    gamma = (x - M / 2) * BW / M
    lambda_ = (1 - x/M) #

    t1_indices = ((n_symbol - 1) * lambda_.unsqueeze(-1)).long()
    mask_t1 = (arange < t1_indices)
    mask_t2 = ~mask_t1
    # TODO: for some reason sign is different from matlab
    shift = torch.exp(1j * 2 * np.pi * (beta * (t_symbol ** 2)) * direction)
    t1 = torch.exp(1j * 2 * np.pi * (t_symbol * gamma.unsqueeze(-1) ) * direction) * shift
    t2 = torch.exp(1j * 2 * np.pi * (t_symbol * (-BW + gamma.unsqueeze(-1)) ) * direction) * shift

    y = t1 * mask_t1 + t2  * mask_t2

    return y.flatten(start_dim  = -2)

def torch_binary(x: Tensor, n_bits: int) -> Tensor:
    """
    Create binary repr of a uint tensor
    """
    dtype = x.dtype
    mask = 2**torch.arange(n_bits).to(x.device, x.dtype)

    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte().float()

def be(x: Tensor, y: Tensor, M) -> Tensor:
    """
    Calculate bit error between two tensors
    TODO: consider this for n_bits <= 8
    https://stackoverflow.com/questions/57871287/pytorch-equivalent-of-numpy-unpackbits
    """
    # Convert to torch tensors
    x_bits = torch_binary(x, M)
    y_bits = torch_binary(y, M)

    # Compute the absolute difference
    bit_diff = torch.abs(x_bits - y_bits)

    # Sum the differences
    be = torch.sum(bit_diff, dim = (-2, -1))

    return be


def signal_convolve(a: Tensor, b:Tensor) -> Tensor:
    """
    Signal convolution of two arrays along last axis
    """
    a1 = a.view(-1, 1, a.shape[-1])
    b1 = b.view(-1, 1, b.shape[-1]).flip(-1)

    # # Calculate padding
    pad = b1.shape[-1] - 1

    a1_padded = F.pad(a1, (pad, pad))

    result = F.conv1d(a1_padded, b1)

    return result


def per(x: Tensor, y: Tensor, N: int, Nm: int) -> Tensor:
    """
    PER Calculation for tensors in PyTorch
    """
    return (((x != y).view(-1, N, Nm)).sum(dim = -1) > 0).sum(dim = -1)/ N


# SER Calculation
def ser(x: Tensor, y:  Tensor, Ns: int) ->  Tensor:
    """
    Ser Calculation
    """
    return (x != y).sum(-1) / Ns
