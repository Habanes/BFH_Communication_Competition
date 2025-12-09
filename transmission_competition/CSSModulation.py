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

    def __init__(self, fs: float, T_symbol: float, f_start: float, bandwidth: float):
        """
        :param fs: sampling frequency [Hz]

        :param T_symbol: Duration of one chirp symbol [s]

        :param f_start: Start frequency of the up-chirp [Hz]

        :param bandwidth: Frequency spanof the chirp [Hz]
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
        self.chip_down = np.cos(phase_down)

    def CSSModulator(self, bits: np.ndarray) -> np.ndarray:
        """
        Map a binary sequence to a concatenation of chip symbols.

        :param bits: np.ndarray
                     1D array (dtype can be int, bool . etc)

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
                signal[start_idx:end_idx] = self.chip_down

        return signal

    def CSSDemodulator(self, rx_signal: np.ndarray) -> np.ndarray:
        """
        :param rx_signal: np.ndarray
                          1D array of time-domain samples (possibly noisy), conataining an integer number of chirp symbols in sequence.

        :return bits_hat: np.ndarray
                          1D array of deteced bits.
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
        down_energy = np.dot(self.chip_down, self.chip_down)

        for i in range(n_symbols):
            start = i * self.Ns
            end = start + self.Ns
            block = rx_signal[start:end]

            # Avoid divisions by zero if block is all zero
            block_energy = np.dot(block, block) + 1e-12

            # Normalised correlation with up- and down chirp
            corr_up = np.dot(block, self.chirp_up) / np.sqrt(up_energy * block_energy)

            corr_down = np.dot(block, self.chip_down) / np.sqrt(
                down_energy * block_energy
            )

            # Decide bit based on which correlation is larger
            if corr_up >= corr_down:
                bits_hat[i] = 0
            else:
                bits_hat[i] = 1
        return bits_hat


def addAWGN(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    :param signal: np.ndarray
                    Transmit signal

    :param snr_db: float
                    Desired signalt to noise ratio in dB.

    :return noisy_signal : ndarray
                            Signal + AWGN noise
    """
    # Linear SNR
    snr_lin = 10.0 ** (snr_db / 10)

    # Signal mean power
    sig_power = np.mean(signal**2)

    # noise Power to thtat SNR= sig_power / noise_power
    noise_power = sig_power + snr_lin

    # AWGN
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))

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
    signal = css.CSSModulator(bits)

    # Modulation
    tx = css.CSSModulator(bits)

    # Füge etwas Rauschen hinzu (optional)
    snr_db = 20.0
    snr_lin = 10 ** (snr_db / 10.0)
    sig_power = np.mean(tx**2)
    noise_power = sig_power / snr_lin
    noise = np.sqrt(noise_power) * np.random.randn(len(tx))
    rx = tx + noise

    # Demodulation
    bits_hat = css.CSSDemodulator(rx)

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
    plt.plot(t, css.chip_down)
    plt.title("Down-Chirp (frequency decreases)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    # === Plot full modulated CSS signal ===
    T_total = len(signal) / fs
    t_full = np.linspace(0, T_total, len(signal), endpoint=False)

    plt.subplot(3, 1, 3)
    plt.plot(t_full, signal)
    plt.title("Modulated CSS Signal for Bit Sequence: " + str(bits))
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

    ### Rausch Test

    fs = 48000.0
    T_symbol = 0.005  # 5 ms pro Chirp
    f_start = 2000.0
    B = 4000.0

    css = CSSModulator(fs, T_symbol, f_start, B)

    bits = np.random.randint(0, 2, size=50)
    bits_hat = addAWGN(css, bits, snr_db=15.0)

    print("Errors:", np.sum(bits != bits_hat))


if __name__ == "__main__":
    main()
