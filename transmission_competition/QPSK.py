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