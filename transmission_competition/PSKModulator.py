import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

class PSKModulator:
    def __init__(self):
        # Parameters needed for both modulation and demodulation
        self.Ac = 1.0           # Carrier Amplitude
        self.fc = 10.0          # Carrier Frequency (Hz)
        self.symbol_rate = 1.0  # 1 Symbol per second
        self.fs = 100.0         # Sampling Rate

        self.duration_per_symbol = 1.0 / self.symbol_rate
        self.sps = int(self.fs * self.duration_per_symbol) # Samples per symbol (100)
    
    # --- NOISE METHOD ---
    def add_awgn_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Adds Additive White Gaussian Noise (AWGN) to the signal based on the desired SNR_dB.
        """
        Ps = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10.0)
        Pn = Ps / snr_linear
        sigma = np.sqrt(Pn)
        noise = sigma * np.random.normal(size=signal.shape)
        return signal + noise
    
    # --- MODULATION ---
    def PSK_modulate(self, bits: np.ndarray):
        
        symbols = bits.reshape(-1, 2)
        phase_array = []

        for b1, b2 in symbols:
            # Map bits to phase (your specific mapping)
            if   b1 == 1 and b2 == 1: phase = np.pi/4
            elif b1 == 1 and b2 == 0: phase = 3*np.pi/4
            elif b1 == 0 and b2 == 1: phase = -np.pi/4
            elif b1 == 0 and b2 == 0: phase = -3*np.pi/4
            phase_array.append(phase)

        phase_array = np.array(phase_array)

        theta_t = np.repeat(phase_array, self.sps)

        total_time = len(theta_t) / self.fs
        t = np.arange(0, total_time, 1/self.fs)
        t = t[:len(theta_t)] # Ensure length match

        signal = self.Ac * np.cos(2 * np.pi * self.fc * t + theta_t)
        
        return signal
        
    # --- DEMODULATION ---
    def PSK_demodulate(self, signal: np.ndarray):
        """
        Performs Coherent IQ Demodulation to recover the original bits.
        """
        # 1. GENERATE LOCAL CARRIERS
        t = np.arange(0, len(signal) / self.fs, 1/self.fs)[:len(signal)]
        LO_I = np.cos(2 * np.pi * self.fc * t)
        LO_Q = -np.sin(2 * np.pi * self.fc * t)

        # 2. MIXING
        I_mixed = signal * LO_I
        Q_mixed = signal * LO_Q

        # 3. LOW PASS FILTERING
        nyquist = 0.5 * self.fs
        cutoff = 2.0  
        b, a = butter(5, cutoff / nyquist, btype='low')
        
        I_filtered = lfilter(b, a, I_mixed) * 2
        Q_filtered = lfilter(b, a, Q_mixed) * 2

        # 4. SAMPLING (Symbol Timing)
        sample_points = np.arange(self.sps // 2, len(signal), self.sps)
        I_sym = I_filtered[sample_points]
        Q_sym = Q_filtered[sample_points]
        
        # 5. DECISION (Mapping Phase to Bits)
        recovered_bits = []
        recovered_phases = np.arctan2(Q_sym, I_sym)

        for angle in recovered_phases:
            if (np.pi/2 > angle >= 0):       # Target: pi/4 -> (1, 1)
                recovered_bits.extend([1, 1])
            elif (np.pi >= angle >= np.pi/2):  # Target: 3pi/4 -> (1, 0)
                recovered_bits.extend([1, 0])
            elif (0 > angle >= -np.pi/2):      # Target: -pi/4 -> (0, 1)
                recovered_bits.extend([0, 1])
            else:                            # Target: -3pi/4 -> (0, 0)
                recovered_bits.extend([0, 0])

        return np.array(recovered_bits)

if __name__ == "__main__":
    # -------------------------------------------------------------------
    # --- PLOTTING CONTROL VARIABLES ---
    # Set the desired length for the time-domain plots (in symbols)
    N_SYMBOLS_TO_PLOT = 4
    # Set the maximum number of symbols for the constellation plot
    CONSTELLATION_SAMPLE_LIMIT = 400
    
    # SYSTEM PARAMETERS
    TEST_SNR_DB = 2.0 # 10 dB Signal-to-Noise Ratio
    TOTAL_TEST_BITS = 800 # Total bits to send for BER calculation (400 symbols)
    # -------------------------------------------------------------------
    
    mod = PSKModulator()
    test_bits = np.random.randint(0, 2, TOTAL_TEST_BITS)
    modulated_signal = mod.PSK_modulate(test_bits)
    noisy_signal = mod.add_awgn_noise(modulated_signal, TEST_SNR_DB)
    demodulated_bits = mod.PSK_demodulate(noisy_signal)

    # --- Verification ---
    errors = np.sum(np.abs(test_bits - demodulated_bits))
    total_bits = len(test_bits)
    
    print("-" * 50)
    print(f"Test Run with SNR = {TEST_SNR_DB} dB")
    print(f"Total Bits Sent: {total_bits}")
    print(f"Total Bit Errors: {errors}")
    print(f"Bit Error Rate (BER): {errors / total_bits:.6f}")
    print("-" * 50)

    # --- Dynamic Plotting Limits ---
    # 1 symbol = mod.sps samples = 2 bits
    N_SAMPLES_PLOT_LIMIT = N_SYMBOLS_TO_PLOT * mod.sps
    T_PLOT_LIMIT = N_SYMBOLS_TO_PLOT * mod.duration_per_symbol 
    N_BITS_PLOT_LIMIT = N_SYMBOLS_TO_PLOT * 2 
    
    # --- Visualization ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1. Modulated signal (Time Domain)
    t_mod = np.arange(len(modulated_signal)) / mod.fs
    axes[0].plot(t_mod[:N_SAMPLES_PLOT_LIMIT], modulated_signal[:N_SAMPLES_PLOT_LIMIT], label='Original Signal', alpha=0.6)
    axes[0].plot(t_mod[:N_SAMPLES_PLOT_LIMIT], noisy_signal[:N_SAMPLES_PLOT_LIMIT], label='Noisy Signal', alpha=0.9)
    axes[0].set_title(f'Signal Comparison (First {N_SYMBOLS_TO_PLOT} Symbols, T={T_PLOT_LIMIT:.2f} s)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Constellation Diagram
    
    # --- Recalculate I/Q for the Constellation Plot (limiting the symbols) ---
    t_plot = np.arange(0, len(noisy_signal) / mod.fs, 1/mod.fs)[:len(noisy_signal)]
    LO_I = np.cos(2 * np.pi * mod.fc * t_plot)
    LO_Q = -np.sin(2 * np.pi * mod.fc * t_plot)
    I_mixed = noisy_signal * LO_I
    Q_mixed = noisy_signal * LO_Q
    
    nyquist = 0.5 * mod.fs
    b, a = butter(5, 2.0 / nyquist, btype='low')
    I_filtered = lfilter(b, a, I_mixed) * 2
    Q_filtered = lfilter(b, a, Q_mixed) * 2
    
    sample_points = np.arange(mod.sps // 2, len(noisy_signal), mod.sps)
    
    # Apply CONSTELLATION_SAMPLE_LIMIT
    I_plot = I_filtered[sample_points][:CONSTELLATION_SAMPLE_LIMIT]
    Q_plot = Q_filtered[sample_points][:CONSTELLATION_SAMPLE_LIMIT]
    
    ideal_I = np.array([np.cos(np.pi/4), np.cos(3*np.pi/4), np.cos(-np.pi/4), np.cos(-3*np.pi/4)])
    ideal_Q = np.array([np.sin(np.pi/4), np.sin(3*np.pi/4), np.sin(-np.pi/4), np.sin(-3*np.pi/4)])
    
    axes[1].plot(I_plot, Q_plot, 'b.', alpha=0.5, label='Received Symbols')
    axes[1].plot(ideal_I, ideal_Q, 'rx', markersize=10, markeredgewidth=2, label='Ideal Symbols')
    axes[1].set_title(f'QPSK Constellation Diagram (SNR={TEST_SNR_DB} dB, {len(I_plot)} Symbols)')
    axes[1].set_xlabel('In-phase (I) Component')
    axes[1].set_ylabel('Quadrature (Q) Component')
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].axvline(0, color='gray', linewidth=0.5)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim([-1.5, 1.5])
    axes[1].set_ylim([-1.5, 1.5])
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. Bit Error Visual
    error_indices = np.where(test_bits != demodulated_bits)[0]
    
    axes[2].stem(range(N_BITS_PLOT_LIMIT), test_bits[:N_BITS_PLOT_LIMIT], basefmt=" ", linefmt='C0-', markerfmt='C0o', label='Sent Bits')
    axes[2].scatter(error_indices[error_indices < N_BITS_PLOT_LIMIT], demodulated_bits[error_indices[error_indices < N_BITS_PLOT_LIMIT]], marker='x', color='red', s=50, label='Errors')
    axes[2].set_title(f'Sent Bits with Error Locations (First {N_BITS_PLOT_LIMIT} Bits)')
    axes[2].set_xlabel('Bit Index')
    axes[2].set_ylabel('Bit Value')
    axes[2].set_ylim([-0.5, 1.5])
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()