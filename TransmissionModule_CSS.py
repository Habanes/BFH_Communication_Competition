"""
File: TransmissionModule_CSS.py
Author: Modified by AI Assistant
Description: Transmission module using CSS Modulation instead of QPSK
"""

from typing import Optional
import numpy as np
import os
from transmission_competition.HuffmanCoder import HuffmanCoder
from transmission_competition.LempelZivCoder import LempelZivCoder
from transmission_competition.HammingCoder74 import HammingCoder74
from transmission_competition.EntropyCalculator import EntropyCalculator
from transmission_competition.CSSModulator import CSSModulator  # Changed from Modulation


class TransmissionModule_CSS:
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
        
        # Initialise CSS modulation class (using default parameters)
        self.modulator = CSSModulator()  # Can customize: CSSModulator(fs=48000.0, T_symbol=0.01, f_start=1000.0, bandwidth=3000.0)
    
        # Source encode the string
        self.source_coded = self.sourceCoder.encode(self.input_string)

        # Channel encode the source_coded string
        self.channel_coded = self.channelCoder.encode(self.source_coded)

        # NOTE: CSS modulation handles 1 bit per symbol, so no padding needed like QPSK
        # Modulate the channel_coded bits using CSS
        self.modulated_signal = self.modulator.CSS_modulate(self.channel_coded)
        
        # Add noise if SNR is specified
        if self.snr_db is not None:
            self.transmitted_signal = self.modulator.add_awgn_noise(self.modulated_signal, self.snr_db)
        else:
            self.transmitted_signal = self.modulated_signal
        
        # Demodulate the transmitted signal
        self.demodulated_bits = self.modulator.CSS_demodulate(self.transmitted_signal)
        
        # Simulate transmission through a noisy channel (bit-level errors)
        self.transmitted, self.error_description = self.channelCoder.channel_simulator(self.demodulated_bits, self.per_bit_error_rate)

        # Channel decode the transmitted string
        self.channel_decoded = self.channelCoder.decode(self.transmitted)

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
        
        return (f"TransmissionModule_CSS:\n"
                f"*****SUMMARY*******************************************************************\n\n"
                f"**  Input String: '{self.input_string}'\n\n"
                f"**  Entropy Calculations: {self.entropyCalculator}\n\n"                
                f"**  Source Coded: \n'{array_to_str(self.source_coded)}'\n\n"
                f"**  Channel Coded: \n'{array_to_str(self.channel_coded)}'\n\n"
                f"**  CSS Modulated Signal: {len(self.modulated_signal)} samples\n\n"
                f"**  SNR (dB): {self.snr_db if self.snr_db is not None else 'N/A (no noise)'}\n\n"
                f"**  Demodulated Bits: \n'{array_to_str(self.demodulated_bits)}'\n\n"
                f"**  Transmitted: \n'{array_to_str(self.transmitted)}'\n\n"
                f"**  Error Description: '{self.error_description}'\n\n"
                f"**  Channel Decoded: \n'{array_to_str(self.channel_decoded)}'\n\n"
                f"**  Output String: '{self.output_string}'\n\n"
                f"**  Lossless: {self.lossless}\n\n"
                f"******************************************************************************\n")
        
        
if __name__ == "__main__":
    print("="*80)
    print("Testing TransmissionModule with CSS Modulation")
    print("="*80)
    
    # Test 1: Simple text without noise
    print("\n--- Test 1: Simple text, no noise ---")
    text1 = "hello,world"
    tm1 = TransmissionModule_CSS(input_string=text1, use_lempel_ziv=False, snr_db=None)
    print(f"Input:  '{tm1.input_string}'")
    print(f"Output: '{tm1.output_string}'")
    print(f"Lossless: {tm1.lossless}")
    
    # Test 2: With noise
    print("\n--- Test 2: Same text with SNR=15dB ---")
    tm2 = TransmissionModule_CSS(input_string=text1, use_lempel_ziv=False, snr_db=15.0)
    print(f"Input:  '{tm2.input_string}'")
    print(f"Output: '{tm2.output_string}'")
    print(f"Lossless: {tm2.lossless}")
    
    # Test 3: With Lempel-Ziv
    print("\n--- Test 3: Using Lempel-Ziv encoding with SNR=10dB ---")
    tm3 = TransmissionModule_CSS(input_string=text1, use_lempel_ziv=True, snr_db=10.0)
    print(f"Input:  '{tm3.input_string}'")
    print(f"Output: '{tm3.output_string}'")
    print(f"Lossless: {tm3.lossless}")
    
    # Test 4: Longer text
    print("\n--- Test 4: Longer text with SNR=12dB ---")
    text4 = "The quick brown fox jumps over the lazy dog"
    tm4 = TransmissionModule_CSS(input_string=text4, use_lempel_ziv=False, snr_db=12.0)
    print(f"Input:  '{tm4.input_string}'")
    print(f"Output: '{tm4.output_string}'")
    print(f"Lossless: {tm4.lossless}")
    print(f"Modulated signal samples: {len(tm4.modulated_signal)}")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
