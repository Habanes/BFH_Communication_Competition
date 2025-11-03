"""
File: TransmissionModule.py
Author: Hannes Stalder
Description: Main transmission module.
"""

from typing import Optional
import numpy as np
from transmission_competition.HuffmanCoder import HuffmanCoder
from transmission_competition.LempelZivCoder import LempelZivCoder
from transmission_competition.HammingCoder74 import HammingCoder74
from transmission_competition.EntropyCalculator import EntropyCalculator


class TransmissionModule:
    def __init__(self, input_string: str, use_lempel_ziv: bool = False, per_bit_error_rate: float = 0.0):
        
        self.input_string: str = input_string
        self.entropyCalculator = EntropyCalculator(self.input_string)
        
        self.per_bit_error_rate = per_bit_error_rate
        
        # Initialise source coder class
        if use_lempel_ziv:
            self.sourceCoder = LempelZivCoder()
        else:
            self.sourceCoder = HuffmanCoder()
            self.sourceCoder.build_encoding_map(input_string)
        
        # Initialise channel coder class
        self.channelCoder = HammingCoder74()
    
        # Source encode the string
        self.source_coded = self.sourceCoder.encode(self.input_string)

        # Channel encode the source_coded string
        self.channel_coded = self.channelCoder.encode(self.source_coded)

        # Simulate transmission through a noisy channel
        self.transmitted, self.error_description = self.channelCoder.channel_simulator(self.channel_coded, self.per_bit_error_rate)

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
                f"**  Transmitted: \n'{array_to_str(self.transmitted)}'\n\n"
                f"**  Error Description: '{self.error_description}'\n\n"
                f"**  Channel Decoded: \n'{array_to_str(self.channel_decoded)}'\n\n"
                f"**  Output String: '{self.output_string}'\n\n"
                f"**  Lossless: {self.lossless}\n\n"
                f"******************************************************************************\n")
        
        
if __name__ == "__main__":
    input_string = "BIG"
    myTransmissionModule = TransmissionModule(input_string=input_string, use_lempel_ziv=True)
    print(myTransmissionModule)