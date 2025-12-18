"""
File: CSSModulation.py
Author: Paranithan Paramalingam
Description: Chirp Spread Spectrum Modulation
"""

import numpy as np


class CSSModulator:
    """
    Simple Chirp Spread Spectrum (CSS) modulator for binary symbols.
    - Bit 0 -> up-chirp
    - Bit 1 -> down-chirp
    """

    """

    # Was der Code als Eingabe erhält
    1) fs – Abtastfrequenz in Hz
    2) T_symbol – Dauer eines Chirp-Symbols (Sekunden)
    3) f_start – Startfrequenz des Chirps
    4) bandwidth – Frequenzbandbreite, über die der Chirp „sweepen“ soll

    # Was der Code genau macht
    1) Berechnet die Anzahl der Samples pro Symbol.
    2) Definiert den Zeitvektor t.
    3) Berechnet die Chirp-Steigung k = B / T.
    4) Erzeugt einen Up-Chirp, dessen Frequenz linear ansteigt.
    5) Erzeugt einen Down-Chirp, dessen Frequenz linear abfällt.
    6) Beide Chirps werden als Kosinus-Signale mit zeitabhängiger Phase erzeugt.

    # Was der Code zurückgibt / bereitstellt
    -  Die Klasse liefert zwei Wellenformen:
    1) self.up_chirp → Signal zur Darstellung eines Bit 0
    2) self.down_chirp → Signal zur Darstellung eines Bit 1
    
    """

    def __init__(self, fs: float = 48000.0, T_symbol: float = 0.01, f_start: float = 1000.0, bandwidth: float = 3000.0):
        """
        :param fs: sampling frequency [Hz] (default: 48000.0)

        :param T_symbol: Duration of one chirp symbol [s] (default: 0.01)

        :param f_start: Start frequency of the up-chirp [Hz] (default: 1000.0)

        :param bandwidth: Frequency span of the chirp [Hz] (default: 3000.0)
        """
        self.fs = fs
        self.Ts = T_symbol
        self.f_start = f_start
        self.B = bandwidth

        # Number of samples per symbol
        self.Ns = int(np.round(self.Ts * self.fs))

        # Time vector for one symbol
        t = np.linspace(0.0, self.Ts, self.Ns, endpoint=False)

        # Chirp rate k  = B/T
        k = self.B / self.Ts

        # Up-chirp: frequency increases from f-start to (f-start + B)
        # Instantenous phase(t) = 2pi (f-start * t + 0.5 * k * t^2)
        phase_up = 2 * np.pi * (self.f_start * t + 0.5 * k * t * t)
        self.chirp_up = np.cos(phase_up)

        # Down-chirp: frequency decreases from (f-start + B) to f-start
        # Instantenous phase(t) = 2pi*[(f-start * B) t - 0.5 * k * t^2)]
        phase_down = 2 * np.pi * ((self.f_start + self.B) * t - 0.5 * k * t * t)
        self.chirp_down = np.cos(phase_down)

    def CSS_modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Map a binary sequence to a concatenation of chirp symbols.

        :param bits: np.ndarray
                     1D array (dtype can be int, bool, etc.)

        Return:
        :param signal : np.ndarray
                        1D array of time-domain samples containing all chirp symbols concatenated in order.
        """

        bit = np.asarray(bits).astype(int)

        # Pre-allocate outüut array: one chirp (NS samples) per bit
        signal = np.zeros(len(bits) * self.Ns, dtype=float)

        for i, b in enumerate(bit):
            start_idx = i * self.Ns
            end_idx = start_idx + self.Ns
            if b == 0:
                signal[start_idx:end_idx] = self.chirp_up
            else:
                signal[start_idx:end_idx] = self.chirp_down

        return signal

    def CSS_demodulate(self, rx_signal: np.ndarray) -> np.ndarray:
        """
        :param rx_signal: np.ndarray
                          1D array of time-domain samples (possibly noisy), containing an integer number of chirp symbols in sequence.

        :return bits_hat: np.ndarray
                          1D array of detected bits.
        """

        """
        1) berechnet, wie viele komplette   Chirps enthalten sind.
        2)Alles abschneiden, was nicht in ganze Chirp-Blöcke passt.
        3) Die Blcöke in ein array laden
        4) Enerigien berechnen (aus Korrelation)
        5) Bit wird anhand von energie demoduliert zurückgegeben
        """

        rx_signal = np.asarray(rx_signal, dtype=float)

        n_symbols = len(rx_signal) // self.Ns

        if n_symbols == 0:
            return np.array([], dtype=int)

        rx_signal = rx_signal[: n_symbols * self.Ns]

        bits_hat = np.zeros(n_symbols, dtype=int)

        # Chirps energies

        up_energy = np.dot(self.chirp_up, self.chirp_up)
        down_energy = np.dot(self.chirp_down, self.chirp_down)

        for i in range(n_symbols):
            start = i * self.Ns
            end = start + self.Ns
            block = rx_signal[start:end]

            # Avoid divisions by zero if block is all zero
            block_energy = np.dot(block, block) + 1e-12

            # Normalised correlation with up- and down chirp
            corr_up = np.dot(block, self.chirp_up) / np.sqrt(up_energy * block_energy)

            corr_down = np.dot(block, self.chirp_down) / np.sqrt(
                down_energy * block_energy
            )

            # Decide bit based on which correlation is larger
            if corr_up >= corr_down:
                bits_hat[i] = 0
            else:
                bits_hat[i] = 1
        return bits_hat
    
    def add_awgn_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Adds Additive White Gaussian Noise (AWGN) to the signal based on the desired SNR_dB.
        
        :param signal: np.ndarray
                        Transmit signal

        :param snr_db: float
                        Desired signal to noise ratio in dB.

        :return noisy_signal : ndarray
                                Signal + AWGN noise
        """
        # Signal mean power
        sig_power = np.mean(signal**2)
        
        # Linear SNR
        snr_lin = 10.0 ** (snr_db / 10.0)
        
        # Noise power such that SNR = sig_power / noise_power
        noise_power = sig_power / snr_lin
        
        # Generate AWGN
        sigma = np.sqrt(noise_power)
        noise = sigma * np.random.randn(len(signal))

        return signal + noise


def main():
    import matplotlib.pyplot as plt

    fs = 48000.0
    T_symbol = 0.01
    f_start = 1000.0
    B = 3000.0

    css = CSSModulator(fs, T_symbol, f_start, B)

    # Test bits
    bits = np.array([0, 1, 0, 0, 1, 1, 0])
    
    # Modulation
    tx = css.CSS_modulate(bits)

    # Add AWGN noise
    snr_db = 20.0
    rx = css.add_awgn_noise(tx, snr_db)

    # Demodulation
    bits_hat = css.CSS_demodulate(rx)

    print("Original bits:    ", bits)
    print("Detected bits:    ", bits_hat)
    print("Bit errors:       ", np.sum(bits != bits_hat))

    # === Plot single up-chirp and down-chirp ===

    t = np.linspace(0, T_symbol, css.Ns, endpoint=False)

    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, css.chirp_up)
    plt.title("Up-Chirp (frequency increases)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.plot(t, css.chirp_down)
    plt.title("Down-Chirp (frequency decreases)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    # === Plot full modulated CSS signal ===
    T_total = len(tx) / fs
    t_full = np.linspace(0, T_total, len(tx), endpoint=False)

    plt.subplot(3, 1, 3)
    plt.plot(t_full, tx)
    plt.title("Modulated CSS Signal for Bit Sequence: " + str(bits))
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

    # === Additional test with noise ===
    print("\n" + "="*50)
    print("Additional noise test:")
    print("="*50)
    
    fs2 = 48000.0
    T_symbol2 = 0.005  # 5 ms per chirp
    f_start2 = 2000.0
    B2 = 4000.0

    css2 = CSSModulator(fs2, T_symbol2, f_start2, B2)

    test_bits = np.random.randint(0, 2, size=50)
    tx2 = css2.CSS_modulate(test_bits)
    rx2 = css2.add_awgn_noise(tx2, snr_db=15.0)
    bits_hat2 = css2.CSS_demodulate(rx2)

    print(f"Total bits: {len(test_bits)}")
    print(f"Errors: {np.sum(test_bits != bits_hat2)}")
    print(f"BER: {np.sum(test_bits != bits_hat2) / len(test_bits):.6f}")


if __name__ == "__main__":
    main()
