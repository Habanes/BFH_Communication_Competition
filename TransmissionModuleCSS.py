""" 
File: TransmissionModuleCSS.py
Author: Felix Egger
Description: Hardware transmission module using Chirp Spread Spectrum (CSS).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from transmission_competition.HuffmanCoder import HuffmanCoder
from transmission_competition.LempelZivCoder import LempelZivCoder
from transmission_competition.HammingCoder74 import HammingCoder74
from transmission_competition.EntropyCalculator import EntropyCalculator
from transmission_competition.CSSModulator import CSSModulator
from transmission_competition.Synchroniser import Synchroniser


def _bits_preview(bits: np.ndarray, max_bits: int = 512) -> str:
    b = np.asarray(bits).astype(int).reshape(-1)
    if b.size <= max_bits:
        return "".join(map(str, b.tolist()))
    head = "".join(map(str, b[: max_bits // 2].tolist()))
    tail = "".join(map(str, b[-max_bits // 2 :].tolist()))
    return f"{head}...{tail} (len={b.size})"


class TransmissionModuleCSS:
    def __init__(
        self,
        input_string: str,
        use_lempel_ziv: bool = True,
        fs: float = 48000.0,
        T_symbol: float = 0.1,
        f_start: float = 1000.0,
        bandwidth: float = 3000.0,
        snr_db: Optional[float] = None,
    ):
        self.input_string: str = input_string
        self.entropyCalculator = EntropyCalculator(self.input_string) if self.input_string else None

        self.use_lempel_ziv = bool(use_lempel_ziv)
        self.fs = float(fs)
        self.snr_db = snr_db

        if self.use_lempel_ziv:
            self.sourceCoder = LempelZivCoder()
        else:
            self.sourceCoder = HuffmanCoder()
            if self.input_string:
                self.sourceCoder.build_encoding_map(input_string)

        self.channelCoder = HammingCoder74()
        self.modulator = CSSModulator(fs=self.fs, T_symbol=T_symbol, f_start=f_start, bandwidth=bandwidth)

        self.synchroniser = Synchroniser(
            preamble_length=int(0.12 * self.fs),
            postamble_length=int(0.12 * self.fs),
            f0=6000.0,
            f1=11000.0,
            sample_rate=self.fs,
        )

        self.hardware = None

        self.source_coded: Optional[np.ndarray] = None
        self.channel_coded: Optional[np.ndarray] = None
        self.modulated_signal: Optional[np.ndarray] = None
        self.transmitted_signal: Optional[np.ndarray] = None

    def build_tx_signal(self) -> np.ndarray:
        self.source_coded = self.sourceCoder.encode(self.input_string)
        self.channel_coded = self.channelCoder.encode(self.source_coded)

        self.modulated_signal = self.modulator.CSS_modulate(self.channel_coded)

        if self.snr_db is not None:
            self.transmitted_signal = self.modulator.add_awgn_noise(self.modulated_signal, self.snr_db)
        else:
            self.transmitted_signal = self.modulated_signal

        return self.synchroniser.pad(self.transmitted_signal)

    def debug_dump_tx_bits(self) -> None:
        if self.source_coded is None or self.channel_coded is None:
            tx = self.build_tx_signal()
        else:
            tx = self.build_tx_signal()

        ns = int(getattr(self.modulator, "Ns", 0))
        ts = float(getattr(self.modulator, "Ts", 0.0))
        n_bits = int(len(self.channel_coded)) if self.channel_coded is not None else 0
        print(f"TX params: fs={self.fs} Ts={ts} Ns={ns} channel_bits={n_bits} expected_payload_s~{n_bits*ts:.2f}s")
        print(f"TX signal: samples={len(tx)} duration_s~{len(tx)/self.fs:.2f}s")
        print(f"TX source_coded_bits: {_bits_preview(self.source_coded)}")
        print(f"TX channel_coded_bits: {_bits_preview(self.channel_coded)}")

    def transmit(self) -> None:
        from transmission_competition.HardwareInterface import HardwareInterface

        if self.hardware is None:
            self.hardware = HardwareInterface(fs=self.fs)
        tx = self.build_tx_signal()
        self.hardware.transmit(tx, blocking=True)

    def decode_from_signal(self, rx_signal: np.ndarray, debug: bool = False) -> str:
        rx_signal = np.asarray(rx_signal, dtype=float)
        pre_len = int(self.synchroniser.preamble_length)
        post_len = int(self.synchroniser.postamble_length)

        if len(rx_signal) >= pre_len + post_len and pre_len > 0 and post_len > 0:
            extracted = rx_signal[pre_len : len(rx_signal) - post_len]
        else:
            extracted, _ = self.synchroniser.depad(rx_signal)

        if debug:
            ns = int(getattr(self.modulator, "Ns", 0))
            ts = float(getattr(self.modulator, "Ts", 0.0))
            print(f"rx_len={len(rx_signal)} extracted_len={len(extracted)} pre={pre_len} post={post_len} fs={self.fs} Ts={ts} Ns={ns}")

        if extracted.size:
            extracted = extracted - float(np.mean(extracted))

            # Fix for clock drift / jitter:
            # If we are slightly short of a full symbol, pad with zeros.
            ns = int(getattr(self.modulator, "Ns", 0))
            if ns > 0:
                remainder = extracted.size % ns
                if remainder > 0:
                    missing = ns - remainder
                    # If we are missing less than 5% of a symbol, pad it.
                    if missing < (0.05 * ns):
                        if debug:
                            print(f"Padding extracted signal with {missing} samples to complete symbol.")
                        extracted = np.pad(extracted, (0, missing), mode='constant')

            f0 = float(getattr(self.modulator, "f_start", 0.0))
            B = float(getattr(self.modulator, "B", 0.0))
            f_low = max(0.0, f0 - 300.0)
            f_high = min(0.5 * self.fs, f0 + B + 300.0)

            X = np.fft.rfft(extracted)
            freqs = np.fft.rfftfreq(len(extracted), d=1.0 / self.fs)
            mask = (freqs >= f_low) & (freqs <= f_high)
            extracted = np.fft.irfft(X * mask, n=len(extracted))

            if debug:
                print(f"bandpass=[{f_low:.1f},{f_high:.1f}]Hz")

        demodulated_bits = self.modulator.CSS_demodulate(extracted)
        demodulated_bits = np.asarray(demodulated_bits, dtype=np.int8)

        if debug:
            print(f"demod_bits={len(demodulated_bits)}")
            if len(demodulated_bits) < 500:
                print("NOTE: Very few symbols detected. If you sent a text file, sender/receiver T_symbol or fs likely mismatch.")
            print(f"RX demod_bits: {_bits_preview(demodulated_bits)}")

        n = (len(demodulated_bits) // 7) * 7
        if n == 0:
            return "Decoding failed: too few bits"
        demodulated_bits = demodulated_bits[:n]

        if debug and len(demodulated_bits) != 0:
            print(f"demod_bits_used={len(demodulated_bits)} (multiple of 7)")

        channel_decoded = self.channelCoder.decode(demodulated_bits)

        if debug:
            print(f"channel_decoded_bits={len(channel_decoded)}")
            print(f"RX channel_decoded_bits: {_bits_preview(channel_decoded)}")

        if len(channel_decoded) < 8:
            return "Decoding failed: too few bits"

        try:
            output_string = self.sourceCoder.decode(channel_decoded)
        except ValueError as e:
            return f"Decoding failed: {str(e)}"
        return output_string

    def receive_and_decode(self, preamble_threshold: float = 0.55, postamble_threshold: float = 0.55, max_listen_s: float = 60.0, debug: bool = False) -> str:
        from transmission_competition.HardwareInterface import HardwareInterface

        if self.hardware is None:
            self.hardware = HardwareInterface(fs=self.fs)

        preamble = self.synchroniser.preamble
        postamble = self.synchroniser.postamble

        # If --force-duration is set, just record for that long and skip preamble/postamble search
        force_duration = None
        if "--force-duration" in sys.argv:
            try:
                idx = sys.argv.index("--force-duration")
                if idx + 1 < len(sys.argv):
                    force_duration = float(sys.argv[idx + 1])
            except ValueError:
                pass

        if force_duration is not None:
            print(f"Forcing recording for {force_duration}s (ignoring preamble/postamble search)...")
            capture = self.hardware.receive_for_duration(duration_s=force_duration)
        else:
            capture = self.hardware.receive_preamble_postamble(
                preamble=preamble,
                postamble=postamble,
                preamble_threshold=preamble_threshold,
                postamble_threshold=postamble_threshold,
                max_listen_s=max_listen_s,
                debug=debug,
            )

        return self.decode_from_signal(capture, debug=debug)

    def record_and_plot(self, duration_s: float = 15.0) -> None:
        from transmission_competition.HardwareInterface import HardwareInterface

        if self.hardware is None:
            self.hardware = HardwareInterface(fs=self.fs)

        signal = self.hardware.receive_for_duration(duration_s=duration_s)

        rms = float(np.sqrt(np.mean(signal.astype(float) ** 2))) if signal.size else 0.0
        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        print(f"Recorded {len(signal)} samples @ {self.fs} Hz")
        print(f"RMS={rms:.6f}  Peak={peak:.6f}")

        import matplotlib.pyplot as plt

        t = np.arange(len(signal)) / self.fs
        plt.figure(figsize=(12, 4))
        plt.plot(t, signal)
        plt.title(f"Recorded microphone signal ({duration_s:.1f}s)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        entropy = self.entropyCalculator if self.entropyCalculator is not None else "N/A"
        return (
            f"TransmissionModuleCSS:\n"
            f"*****SUMMARY*******************************************************************\n\n"
            f"**  Input String: '{self.input_string}'\n\n"
            f"**  Entropy Calculations: {entropy}\n\n"
            f"**  Sample Rate (Hz): {self.fs}\n\n"
            f"**  Coding: {'Lempel-Ziv' if self.use_lempel_ziv else 'Huffman'} + Hamming(7,4)\n\n"
            f"******************************************************************************\n"
        )


if __name__ == "__main__":
    def _get_arg(name: str, default=None):
        if name in sys.argv:
            i = sys.argv.index(name)
            if i + 1 < len(sys.argv):
                return sys.argv[i + 1]
        return default

    fs = float(_get_arg("--fs", 48000.0))
    T_symbol = float(_get_arg("--ts", 0.1))
    f_start = float(_get_arg("--f-start", 1000.0))
    bandwidth = float(_get_arg("--bw", 3000.0))

    role = input("Sender or receiver? [s/r]: ").strip().lower()

    if role.startswith("s"):
        if len(sys.argv) < 2:
            print("Usage: python TransmissionModuleCSS.py <textfile>")
            raise SystemExit(1)

        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            raise SystemExit(1)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tm = TransmissionModuleCSS(
            input_string=text,
            use_lempel_ziv=True,
            fs=fs,
            T_symbol=T_symbol,
            f_start=f_start,
            bandwidth=bandwidth,
        )
        print(tm)

        coded_bits = len(tm.channelCoder.encode(tm.sourceCoder.encode(text)))
        est_payload_s = coded_bits * T_symbol
        print(f"CONFIG: fs={fs} Ts={T_symbol} f_start={f_start} bw={bandwidth}")
        print(f"TX: coded_bits={coded_bits} est_payload_s~{est_payload_s:.1f}s")
        if "--debug" in sys.argv:
            tm.debug_dump_tx_bits()
        tm.transmit()

    else:
        tm = TransmissionModuleCSS(
            input_string="",
            use_lempel_ziv=True,
            fs=fs,
            T_symbol=T_symbol,
            f_start=f_start,
            bandwidth=bandwidth,
        )
        if "--test-record" in sys.argv:
            tm.record_and_plot(duration_s=15.0)
            raise SystemExit(0)

        print(f"CONFIG: fs={fs} Ts={T_symbol} f_start={f_start} bw={bandwidth}")
        
        pre_thresh = float(_get_arg("--pre-thresh", 0.55))
        post_thresh = float(_get_arg("--post-thresh", 0.55))
        print(f"THRESHOLDS: pre={pre_thresh} post={post_thresh}")
        
        decoded = tm.receive_and_decode(
            preamble_threshold=pre_thresh,
            postamble_threshold=post_thresh,
            debug=("--debug" in sys.argv)
        )
        print("\n*****RECEIVED MESSAGE*********************************************************\n")
        print(decoded)
        print("\n******************************************************************************\n")
