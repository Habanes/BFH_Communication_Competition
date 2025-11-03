"""
File: HammingCoder74.py
Author: Hannes Stalder
Description: Implements Hamming(7,4) error correction code for channel coding.
"""

import numpy as np
import random

class HammingCoder74:
    def __init__(self):
        # Matrix used to encode a set of four bits
        self.encoder_matrix = np.array([
            [1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, 1]
        ], dtype=int)
        
        # Matrix used to generate the correction vector
        self.correction_matrix = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ], dtype=int)
        
        # Matrix used to decode the 7 bit coding back to the four bits
        self.decoder_matrix = np.array([
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=int)
    
    # Encodes a n size np array of zeroes and ones to a channel coded version
    def encode(self, data: np.ndarray) -> np.ndarray:
        
        # Pad the array to be a multiple of 4
        padded_len = (len(data) + 3) // 4 * 4
        if len(data) < padded_len:
            data = np.pad(data, (0, padded_len - len(data)), constant_values=0)

        # Encode each set of four bits using the encoding matrix
        encoded_bits = []
        for i in range(0, len(data), 4):
            chunk = data[i:i+4]
            encoded_chunk = np.matmul(chunk, self.encoder_matrix) % 2
            encoded_bits.extend(encoded_chunk)
        
        return np.array(encoded_bits, dtype=np.int8)

    # simulates channel errors by randomly flipping bits based on a specified error rate
    def channel_simulator(self, code: np.ndarray, per_bit_error_rate: float):
        transmitted = code.copy()
        error_positions = []
        
        for i in range(len(transmitted)):
            if random.random() < per_bit_error_rate:
                transmitted[i] = 1 - transmitted[i]  # Flip the bit
                error_positions.append(i)
        
        if not error_positions:
            desc = "No error introduced"
        elif len(error_positions) == 1:
            desc = f"Single bit error at position {error_positions[0]}"
        else:
            desc = f"{len(error_positions)} bit errors at positions {', '.join(map(str, sorted(error_positions)))}"
            
        return transmitted, desc
    
    # decodes a channel coded array back to source coding and corrects errors if able
    def decode(self, received: np.ndarray) -> np.ndarray:
        
        # our encode will always create an array with a length which is divisible by 7
        if len(received) % 7 != 0:
            raise ValueError("Received array length must be a multiple of 7")

        # we go through each set of 7 bits
        decoded_bits = []
        for i in range(0, len(received), 7):
            chunk = received[i:i+7]
            
            # Correct the set of 7 bits if able
            syndrome = np.matmul(self.correction_matrix, chunk) % 2
            position = syndrome[0] * 1 + syndrome[1] * 2 + syndrome[2] * 4
            corrected_chunk = chunk.copy()
            # if the resulting position is zero, there is no error
            if position != 0:
                corrected_chunk[position - 1] = (corrected_chunk[position - 1] + 1) % 2
            
            # Decode back into sets of four bits
            decoded_chunk = np.matmul(corrected_chunk, self.decoder_matrix.T)
            decoded_bits.extend(decoded_chunk)
            
        return np.array(decoded_bits, dtype=np.int8)