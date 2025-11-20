"""
File: QPSK.py
Author: Felix Egger
Description: Implements QPSK modulation and demodulation functions.
"""

import numpy as np

class QPSK:
    def __init__(self):
        
        self.mapping = {
            (0,0): (1+1j)/np.sqrt(2),
            (0,1): (-1+1j)/np.sqrt(2),
            (1,1): (-1-1j)/np.sqrt(2),
            (1,0): (1-1j)/np.sqrt(2)
        }
        
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
    
    def Modulation(self, bit_pairs: np.ndarray) -> np.ndarray:
        
        Symbols = np.array([self.mapping[tuple(pair)] for pair in bit_pairs])
        
        return np.array(Symbols)
    
    def Demodulation(self, received_symbols: np.ndarray) -> np.ndarray:
        
        demodulated_bits = []
        for r in received_symbols:
            distances = {symbol: np.abs(r - mapped_symbol) for symbol, mapped_symbol in self.mapping.items()}
            closest_symbol = min(distances, key=distances.get)
            demodulated_bits.append(closest_symbol)
        
        return np.array(demodulated_bits)