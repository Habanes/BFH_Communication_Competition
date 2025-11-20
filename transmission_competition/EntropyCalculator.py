"""
File: EntropyCalculator.py
Author: Hannes Stalder
Description: Calculates entropy statistics for text.
"""

import numpy as np
from collections import Counter

class EntropyCalculator:
    def __init__(self, text: str):
        if not text:
            raise ValueError("Input string cannot be empty")

        counts = np.array(list(Counter(text).values()), dtype=float)
        p = counts / counts.sum()

        self.H = np.dot(p, np.log2(1 / p))                # entropy
        self.H0 = np.log2(len(p))                         # max entropy
        self.R = self.H0 - self.H                         # absolute redundancy
        self.r = self.R / self.H0 if self.H0 > 0 else 0.0 # relative redundancy

    def __repr__(self):
        return (f"EntropyCalculator(H={self.H:.4f} bits/char, "
                f"H0={self.H0:.4f} bits/char, "
                f"R={self.R:.4f} bits/char, "
                f"r={self.r:.2%})")


# Example usage
if __name__ == "__main__":
    calc = EntropyCalculator("abbbc")
    print(calc)
