import numpy as np

def loramod(x: np.array, SF: int, BW: float, fs:float, direction: int = 1):
    """
    Adapted From: https://github.com/bhomssi/LoRaBERMatlab/tree/main
    TODO: fix sensibility to scale of SF, beta and gamma (precision problems)
    """
    M = 2 ** SF
    if not np.issubdtype(x.dtype, np.integer) or np.any(x < 0) or np.any(x >= M):
        raise ValueError("Input symbols must be integers in the range [0, M-1]")

    Ts = (2**int(SF)) / int(BW)
    beta = BW/(2*Ts)
    n_symbol = fs * M / BW
    arange = np.arange(n_symbol)
    t_symbol = arange[..., None, :]  * 1/fs
    gamma = (x - M / 2) * BW / M
    lambda_ = (1 - x/M) #

    t1_indices = ((n_symbol - 1) * lambda_[..., None]).astype(int)
    mask_t1 = (arange < t1_indices)
    mask_t2 = ~mask_t1
    # TODO: for some reason sign is different from matlab
    shift = np.exp(1j * 2 * np.pi * (beta * (t_symbol ** 2)) * direction)
    t1 = np.exp(1j * 2 * np.pi * (t_symbol * gamma[..., None] ) * direction) * shift
    t2 = np.exp(1j * 2 * np.pi * (t_symbol * (-BW + gamma[..., None]) ) * direction) * shift


    y = t1 * mask_t1 + t2  * mask_t2

    return y.reshape(*y.shape[:-2], -1)


def numpy_binary(x: np.array, n_bits: int) -> np.array:
  """
  Create binary repr of uint array
  """
  if n_bits <= 8:
    bin = np.unpackbits(x[..., None].astype(np.uint8), axis=-1)
    bin = bin[..., n_bits-8:].astype(float)
  else:
    mask = 2**np.arange(n_bits)
    bin = (np.bitwise_and(x[..., None], mask) != 0).astype(np.uint8).astype(float)
  return bin

def be(x: np.array, y: np.array, M) -> np.array:
    """
    Calculate bit error between two arrays
    """
    x_bits = numpy_binary(x, M)
    y_bits = numpy_binary(y, M)

    # Compute the absolute difference
    bit_diff = np.abs(x_bits - y_bits)

    # Sum the differences
    be = np.sum(bit_diff, axis = (-2, -1))

    return be


def per(x: np.array, y: np.array, N: int, Nm: int) -> np.array:
    """
    PER Calculation for batched arrays in NumPy
    """
    z = (x != y).reshape(-1, N, Nm)
    per_batch = np.sum(np.sum(z, axis=-1) > 0, axis=-1) / N
    return per_batch


def ser(x: np.array, y:  np.array, Ns: int) ->  np.array:
    """
    SER Calculation for batched arrays in NumPy
    """
    return (x != y).sum(-1) / Ns