"""
File: EntropyCalculator.py
Author: Hannes Stalder
Description: Calculates entropy statistics for text.
"""

import numpy as np
from collections import Counter

def calculate_entropy_stats(text: str):
    if not text:
        raise ValueError("Input string cannot be empty")

    counts = np.array(list(Counter(text).values()), dtype=float)
    p = counts / counts.sum()

    H = np.dot(p, np.log2(1 / p))                # entropy
    H0 = np.log2(len(p))                         # max entropy
    R = H0 - H                                   # absolute redundancy
    r = R / H0 if H0 > 0 else 0.0                # relative redundancy

    return H, H0, R, r


# Example usage
if __name__ == "__main__":
    H, H0, R, r = calculate_entropy_stats("aabbccdd")
    print(f"H  = {H:.4f} bits/char")
    print(f"H0 = {H0:.4f} bits/char")
    print(f"R  = {R:.4f} bits/char")
    print(f"r  = {r:.2%}")
