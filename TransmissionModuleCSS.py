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
            f0=600.0,
            f1=9000.0,
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

    def transmit(self) -> None:
        from transmission_competition.HardwareInterface import HardwareInterface

        if self.hardware is None:
            self.hardware = HardwareInterface(fs=self.fs)
        tx = self.build_tx_signal()
        self.hardware.transmit(tx, blocking=True)

    def decode_from_signal(self, rx_signal: np.ndarray) -> str:
        extracted, _ = self.synchroniser.depad(np.asarray(rx_signal, dtype=float))

        demodulated_bits = self.modulator.CSS_demodulate(extracted)
        demodulated_bits = np.asarray(demodulated_bits, dtype=np.int8)

        n = (len(demodulated_bits) // 7) * 7
        if n > 0:
            demodulated_bits = demodulated_bits[:n]

        channel_decoded = self.channelCoder.decode(demodulated_bits)

        output_string = self.sourceCoder.decode(channel_decoded)
        return output_string

    def receive_and_decode(self, preamble_threshold: float = 0.70, postamble_threshold: float = 0.70, max_listen_s: float = 60.0) -> str:
        from transmission_competition.HardwareInterface import HardwareInterface

        if self.hardware is None:
            self.hardware = HardwareInterface(fs=self.fs)

        preamble = self.synchroniser.preamble
        postamble = self.synchroniser.postamble

        capture = self.hardware.receive_preamble_postamble(
            preamble=preamble,
            postamble=postamble,
            preamble_threshold=preamble_threshold,
            postamble_threshold=postamble_threshold,
            max_listen_s=max_listen_s,
        )

        return self.decode_from_signal(capture)

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

        tm = TransmissionModuleCSS(input_string=text, use_lempel_ziv=True)
        print(tm)
        tm.transmit()

    else:
        tm = TransmissionModuleCSS(input_string="", use_lempel_ziv=True)
        decoded = tm.receive_and_decode()
        print("\n*****RECEIVED MESSAGE*********************************************************\n")
        print(decoded)
        print("\n******************************************************************************\n")
