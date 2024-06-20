import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from utils.utils_numpy import loramod, be, per, ser

if __name__ == '__main__':
    # Parameters
    bs = 32 # batch size
    # Parameters
    Ns = int(1e2)  # Number of symbols
    Nm = int(1e1)  # Symbols per packet
    N = int(Ns / Nm)

    SF = 4  # Spreading Factor
    M = 2 ** SF  # Modulation levels
    Ts = M  # Symbol duration
    BW = 1  # Bandwidth

    # Time vectors
    t = np.arange(0, 1, 1 / Ts)

    # Generate random symbols
    x = np.random.randint(0, M, (bs, int(Ns)))

    # Create Chirps
    upChirp = loramod(x, SF, BW, BW)
    zero = np.zeros((1), dtype = np.int64)
    dnChirp = loramod(zero, SF, BW, BW, direction=-1)
    dnChirp = np.tile(dnChirp, reps = int(upChirp.shape[1]/dnChirp.shape[0]))
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
        n = (sigma / np.sqrt(2)) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        # Generate random complex numbers
        h = ((np.random.randn(bs, int(Ns / Nm)) + 1j * np.random.randn(bs, int(Ns / Nm))) / np.sqrt(2))
        h = np.abs(np.repeat(h, repeats = M * int(Nm), axis = -1))

        # Received signal
        Rx1 = signal  + n
        Rx2 = h * signal  + n

        idx1 = np.argmax(np.abs(np.fft.fft(Rx1.reshape(bs, int(Ns), M) , axis = -1)), axis = -1) 
        ynCoh1 = idx1
        idx2 = np.argmax(np.abs(np.fft.fft(Rx2.reshape(bs, int(Ns),M) , axis = -1)), axis = -1) 
        ynCoh2 = idx2

        r1 = np.zeros((bs, M, int(Ns)))
        r2 = np.zeros((bs, M, int(Ns)))
        exp_term = np.exp(-1j * 2 * np.pi * (M - np.arange(M))[..., None] * t)

        for i, term in enumerate(exp_term):
            rtemp1 = fftconvolve(Rx1, term[None, :], mode='full', axes = -1)
            r1[:, i] = rtemp1[...,  Ts::Ts].real
            rtemp2 = fftconvolve(Rx2, term[None, :], mode='full', axes = -1)
            r2[:, i] = rtemp2[...,  Ts::Ts].real

        idx = np.argmax(r1, axis=1)
        yCoh1 = idx

        idx = np.argmax(r2, axis=1)
        yCoh2 = idx

        BER_nCOH_AWGN_SIMULATION.append(be(x, ynCoh1, M).mean()/(SF * Ns))
        BER_nCOH_RAY_SIMULATION.append(be(x, ynCoh2, M).mean()/(SF * Ns))
        BER_COH_AWGN_SIMULATION.append(be(x, yCoh1, M).mean()/(SF * Ns))
        BER_COH_RAY_SIMULATION.append(be(x, yCoh2, M).mean()/(SF * Ns))

        SER_nCOH_AWGN_SIMULATION.append(ser(x, ynCoh1, Ns).mean())
        SER_nCOH_RAY_SIMULATION.append(ser(x, ynCoh2, Ns).mean())
        SER_COH_AWGN_SIMULATION.append(ser(x, yCoh1,  Ns).mean())
        SER_COH_RAY_SIMULATION.append(ser(x,yCoh2,  Ns).mean())


        PER_nCOH_AWGN_SIMULATION.append(per(x, ynCoh1, int(N), int(Nm)).mean())
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
