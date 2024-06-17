import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.utils_pytorch import loramod, signal_convolve, be, per, ser 


if __name__ == '__main__':    
    torch.set_default_dtype(torch.float64)
    # Parameters Torch
    device = 'cpu'
    bs = 32 # batch size

    # Parameters LORA
    Ns = int(1e2)  # Number of symbols
    Nm = int(1e1)  # Symbols per packet
    N = int(Ns / Nm)

    SF = 4  # Spreading Factor
    M = 2 ** SF  # Modulation levels
    Ts = M  # Symbol duration
    BW = 1  # Bandwidth

    # Time vectors
    t = torch.arange(0, 1, 1 / Ts, dtype=torch.float64)

    # Generate random symbols
    x = torch.randint(0, M, (bs, int(Ns)), device = device)

    # Create Chirps

    upChirp = loramod(x, SF, BW, BW)
    zero = torch.zeros((1) ,device = x.device, dtype = torch.long)
    dnChirp = loramod(zero, SF, BW, BW, direction=-1)
    dnChirp = dnChirp.repeat(int(upChirp.size(1)/dnChirp.size(0)))

    signal = upChirp * dnChirp

    # SNR Initialization
    SNR = np.arange(-30, 31)
    EbNo = SNR + 10 * np.log10(M / SF)
    EsNo = EbNo + 10 * np.log10(SF)

    # Simulation Results Storage
    BER_nCOH_AWGN_SIMULATION = []
    BER_nCOH_RAY_SIMULATION = []
    BER_COH_AWGN_SIMULATION = []
    BER_COH_RAY_SIMULATION = []

    SER_nCOH_AWGN_SIMULATION = []
    SER_nCOH_RAY_SIMULATION = []
    SER_COH_AWGN_SIMULATION = []
    SER_COH_RAY_SIMULATION = []

    PER_nCOH_AWGN_SIMULATION = []
    PER_nCOH_RAY_SIMULATION = []
    PER_COH_AWGN_SIMULATION = []
    PER_COH_RAY_SIMULATION = []

    # Simulation Loop
    for snr in SNR:

        # SNR to noise sigma
        sigma = 1 / 10 **(snr/ 20.0)

        # AWGN noise
        n = (sigma / np.sqrt(2)) * (torch.randn_like(signal) + 1j * torch.randn_like(signal))
        # Generate random complex numbers
        h = ((torch.randn(bs, int(Ns / Nm), dtype = torch.float, device = x.device) + 1j * torch.randn(bs, int(Ns / Nm), dtype = torch.float, device = x.device)) / np.sqrt(2))
        h = h.repeat((1, M * int(Nm))).abs()

        # Received signal
        Rx1 = signal  + n
        Rx2 = h * signal  + n

        idx1 = torch.argmax(torch.abs(torch.fft.fft(Rx1.reshape(bs, int(Ns), M) , dim = -1)), dim = -1) # matlab takes along the first non 1 dim
        ynCoh1 = idx1
        idx2 = torch.argmax(torch.abs(torch.fft.fft(Rx2.reshape(bs, int(Ns),M) , dim = -1)), dim = -1) # matlab takes along the first non 1 dim
        ynCoh2 = idx2

        exp_term = torch.exp(-1j * 2 * np.pi * (M - torch.arange(M, dtype = torch.float, device = x.device)).unsqueeze(-1) * t)
        rtemp1 = signal_convolve(Rx1, exp_term)
        r1 = rtemp1[...,  Ts::Ts].real
        rtemp2 = signal_convolve(Rx2, exp_term)
        r2 = rtemp2[...,  Ts::Ts].real

        idx = torch.argmax(r1, dim=1)
        yCoh1 = idx

        idx = torch.argmax(r2, dim=1)
        yCoh2 = idx

        BER_nCOH_AWGN_SIMULATION.append(be(x, ynCoh1, M).mean()/(SF * Ns))
        BER_nCOH_RAY_SIMULATION.append(be(x, ynCoh2, M).mean()/(SF * Ns))
        BER_COH_AWGN_SIMULATION.append(be(x, yCoh1, M).mean()/(SF * Ns))
        BER_COH_RAY_SIMULATION.append(be(x, yCoh2, M).mean()/(SF * Ns))


        SER_nCOH_AWGN_SIMULATION.append(ser(x, ynCoh1, Ns).mean())
        SER_nCOH_RAY_SIMULATION.append(ser(x, ynCoh2, Ns).mean())
        SER_COH_AWGN_SIMULATION.append(ser(x, yCoh1,  Ns).mean())
        SER_COH_RAY_SIMULATION.append(ser(x,yCoh2,  Ns).mean())


        PER_nCOH_AWGN_SIMULATION.append(per(x, ynCoh1,  int(N), int(Nm)).mean())
        PER_nCOH_RAY_SIMULATION.append(per(x,ynCoh2,  int(N), int(Nm)).mean())
        PER_COH_AWGN_SIMULATION.append(per(x, yCoh1,  int(N), int(Nm)).mean())
        PER_COH_RAY_SIMULATION.append(per(x, yCoh2,  int(N), int(Nm)).mean())

    # Plotting the results
    plt.figure()
    plt.plot(EbNo, BER_nCOH_AWGN_SIMULATION, 'bs', label='Non-Coherent AWGN')
    plt.plot(EbNo, BER_nCOH_RAY_SIMULATION, 'rs', label='Non-Coherent Rayleigh')
    plt.plot(EbNo, BER_COH_AWGN_SIMULATION, 'go', label='Coherent AWGN')
    plt.plot(EbNo, BER_COH_RAY_SIMULATION, 'ko', label='Coherent Rayleigh')

    plt.xlabel('EbNo (dB)')
    plt.ylabel('BER')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()

    plt.figure()
    plt.plot(SNR, PER_nCOH_AWGN_SIMULATION, 'bs', label='Non-Coherent AWGN')
    plt.plot(SNR, PER_nCOH_RAY_SIMULATION, 'rs', label='Non-Coherent Rayleigh')
    plt.plot(SNR, PER_COH_AWGN_SIMULATION, 'go', label='Coherent AWGN')
    plt.plot(SNR, PER_COH_RAY_SIMULATION, 'ko', label='Coherent Rayleigh')

    plt.xlabel('SNR (dB)')
    plt.ylabel('PER')
    plt.legend()
    plt.grid()
    plt.show()


    plt.figure()
    plt.plot(EsNo, SER_nCOH_AWGN_SIMULATION, 'bs', label='Non-Coherent AWGN')
    plt.plot(EsNo, SER_nCOH_RAY_SIMULATION, 'rs', label='Non-Coherent Rayleigh')
    plt.plot(EsNo, SER_COH_AWGN_SIMULATION, 'go', label='Coherent AWGN')
    plt.plot(EsNo, SER_COH_RAY_SIMULATION, 'ko', label='Coherent Rayleigh')

    plt.xlabel('SNR (dB)')
    plt.ylabel('SER')
    plt.legend()
    plt.grid()
    plt.show()