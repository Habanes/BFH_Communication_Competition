"""
File: TransmissionModule.py
Author: Hannes Stalder
Description: Main transmission module.
"""

from typing import Optional
import numpy as np
import os
from transmission_competition.HuffmanCoder import HuffmanCoder
from transmission_competition.LempelZivCoder import LempelZivCoder
from transmission_competition.HammingCoder74 import HammingCoder74
from transmission_competition.EntropyCalculator import EntropyCalculator
from transmission_competition.PSKModulator import PSKModulator


class TransmissionModule:
    def __init__(self, input_string: str, use_lempel_ziv: bool = False, per_bit_error_rate: float = 0.00, snr_db: Optional[float] = None):
        
        self.input_string: str = input_string
        self.entropyCalculator = EntropyCalculator(self.input_string)
        
        self.per_bit_error_rate = per_bit_error_rate
        self.snr_db = snr_db
        
        # Initialise source coder class
        if use_lempel_ziv:
            self.sourceCoder = LempelZivCoder()
        else:
            self.sourceCoder = HuffmanCoder()
            self.sourceCoder.build_encoding_map(input_string)
        
        # Initialise channel coder class
        self.channelCoder = HammingCoder74()
        
        # Initialise modulation class
        self.modulator = PSKModulator()
    
        # Source encode the string
        self.source_coded = self.sourceCoder.encode(self.input_string)

        # Channel encode the source_coded string
        self.channel_coded = self.channelCoder.encode(self.source_coded)

        # Pad channel_coded to even length for QPSK (needs pairs of bits)
        if len(self.channel_coded) % 2 != 0:
            self.channel_coded_padded = np.append(self.channel_coded, 0)
            self.padding_added = True
        else:
            self.channel_coded_padded = self.channel_coded
            self.padding_added = False

        # Modulate the channel_coded bits
        self.modulated_signal = self.modulator.QPSK_modulate(self.channel_coded_padded)
        
        # Add noise if SNR is specified
        if self.snr_db is not None:
            self.transmitted_signal = self.modulator.add_awgn_noise(self.modulated_signal, self.snr_db)
        else:
            self.transmitted_signal = self.modulated_signal
        
        # Demodulate the transmitted signal
        self.demodulated_bits = self.modulator.QPSK_demodulate(self.transmitted_signal)
        
        # Remove padding if it was added
        if self.padding_added:
            self.demodulated_bits = self.demodulated_bits[:-1]

        # Simulate transmission through a noisy channel (bit-level errors)
        self.transmitted, self.error_description = self.channelCoder.channel_simulator(self.demodulated_bits, self.per_bit_error_rate)

        # Channel decode the transmitted string
        self.channel_decoded_padded = self.channelCoder.decode(self.transmitted)
        
        # For Huffman, the decode handles the exact message length, so no need to trim
        self.channel_decoded = self.channel_decoded_padded

        # Decode the source encoding
        try:
            self.output_string = self.sourceCoder.decode(self.channel_decoded)
        except ValueError as e:
            self.output_string = f"Decoding failed: {str(e)}"
            
        self.lossless = self.input_string == self.output_string

    def __repr__(self):
        # Helper function to convert numpy array to bitstring for display
        def array_to_str(arr):
            if isinstance(arr, np.ndarray):
                return ''.join(map(str, arr.astype(int)))
            return str(arr)
        
        return (f"TransmissionModule:\n"
                f"*****SUMMARY*******************************************************************\n\n"
                f"**  Input String: '{self.input_string}'\n\n"
                f"**  Entropy Calculations: {self.entropyCalculator}\n\n"                
                f"**  Source Coded: \n'{array_to_str(self.source_coded)}'\n\n"
                f"**  Channel Coded: \n'{array_to_str(self.channel_coded)}'\n\n"
                f"**  Modulated Signal: {len(self.modulated_signal)} samples\n\n"
                f"**  SNR (dB): {self.snr_db if self.snr_db is not None else 'N/A (no noise)'}\n\n"
                f"**  Demodulated Bits: \n'{array_to_str(self.demodulated_bits)}'\n\n"
                f"**  Transmitted: \n'{array_to_str(self.transmitted)}'\n\n"
                f"**  Error Description: '{self.error_description}'\n\n"
                f"**  Channel Decoded: \n'{array_to_str(self.channel_decoded)}'\n\n"
                f"**  Output String: '{self.output_string}'\n\n"
                f"**  Lossless: {self.lossless}\n\n"
                f"******************************************************************************\n")
        
        
if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "transmission_competition", "input_text_short.txt")
    file_handle = open(file_path, "r")
    text = file_handle.read()
    file_handle.close()
    
    text = "hello,world"

    myTransmissionModule = TransmissionModule(input_string=text, use_lempel_ziv=True, snr_db=10.0)
    print(myTransmissionModule)