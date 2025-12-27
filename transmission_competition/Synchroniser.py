import numpy as np
from typing import Tuple


class Synchroniser:
    """
    Synchronisation class for adding and removing padding to/from signals.
    Uses chirp signals (frequency ramps) as preamble and postamble for synchronization.
    """
    
    def __init__(self, preamble_length: int = 1000, postamble_length: int = 1000,
                 f0: float = 100, f1: float = 20000, sample_rate: float = 44100):
        """
        Initialize the Synchronisation class.
        
        Parameters:
        -----------
        preamble_length : int
            Length of the preamble in samples (default: 1000)
        postamble_length : int
            Length of the postamble in samples (default: 1000)
        f0 : float
            Starting frequency for chirp in Hz (default: 100)
        f1 : float
            Ending frequency for chirp in Hz (default: 1000)
        sample_rate : float
            Sample rate in Hz (default: 44100)
        """
        self.preamble_length = preamble_length
        self.postamble_length = postamble_length
        self.f0 = f0
        self.f1 = f1
        self.sample_rate = sample_rate
        
        # Generate the preamble and postamble chirps
        self.preamble = self._generate_chirp(self.preamble_length, self.f0, self.f1)
        self.postamble = self._generate_chirp(self.postamble_length, self.f1, self.f0)  # Down-chirp
    
    def _generate_chirp(self, length: int, f_start: float, f_end: float) -> np.ndarray:
        """
        Generate a linear chirp signal.
        
        Parameters:
        -----------
        length : int
            Length of the chirp in samples
        f_start : float
            Starting frequency in Hz
        f_end : float
            Ending frequency in Hz
            
        Returns:
        --------
        np.ndarray
            The generated chirp signal
        """
        t = np.arange(length) / self.sample_rate
        # Linear chirp: frequency changes linearly from f_start to f_end
        k = (f_end - f_start) / (length / self.sample_rate)  # Chirp rate
        phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
        chirp = np.sin(phase)
        
        # Apply window to reduce edge effects
        window = np.hanning(length)
        chirp = chirp * window
        
        return chirp
    
    def pad(self, signal: np.ndarray) -> np.ndarray:
        """
        Add preamble and postamble to the signal for synchronization.
        
        Parameters:
        -----------
        signal : np.ndarray
            The input signal to be padded
            
        Returns:
        --------
        np.ndarray
            The padded signal with preamble and postamble
        """
        padded_signal = np.concatenate([self.preamble, signal, self.postamble])
        return padded_signal
    
    def depad(self, received_signal: np.ndarray, correlation_threshold: float = 0.5) -> Tuple[np.ndarray, dict]:
        """
        Remove preamble and postamble from the received signal using autocorrelation.
        
        Parameters:
        -----------
        received_signal : np.ndarray
            The received signal with preamble and postamble
        correlation_threshold : float
            Threshold for correlation peak detection (0 to 1, default: 0.5)
            
        Returns:
        --------
        Tuple[np.ndarray, dict]
            - The extracted signal without padding
            - Dictionary with synchronization information (preamble_idx, postamble_idx, correlation values)
        """
        # Normalize the preamble and postamble for correlation
        preamble_norm = self.preamble / np.linalg.norm(self.preamble)
        postamble_norm = self.postamble / np.linalg.norm(self.postamble)
        
        # Cross-correlate with preamble to find start
        preamble_correlation = np.correlate(received_signal, preamble_norm, mode='valid')
        preamble_correlation = np.abs(preamble_correlation)
        preamble_idx = np.argmax(preamble_correlation)
        preamble_corr_value = preamble_correlation[preamble_idx]
        
        # Cross-correlate with postamble to find end
        postamble_correlation = np.correlate(received_signal, postamble_norm, mode='valid')
        postamble_correlation = np.abs(postamble_correlation)
        postamble_idx = np.argmax(postamble_correlation)
        postamble_corr_value = postamble_correlation[postamble_idx]
        
        # The actual signal starts after the preamble
        signal_start = preamble_idx + self.preamble_length
        signal_end = postamble_idx
        
        # Extract the signal
        if signal_start < signal_end and signal_start >= 0 and signal_end <= len(received_signal):
            extracted_signal = received_signal[signal_start:signal_end]
        else:
            # If synchronization failed, return the whole signal
            extracted_signal = received_signal
            print(f"Warning: Synchronization may have failed. signal_start={signal_start}, signal_end={signal_end}")
        
        # Return synchronization info
        sync_info = {
            'preamble_idx': preamble_idx,
            'postamble_idx': postamble_idx,
            'signal_start': signal_start,
            'signal_end': signal_end,
            'preamble_correlation': preamble_corr_value,
            'postamble_correlation': postamble_corr_value,
            'max_preamble_corr': np.max(preamble_correlation),
            'max_postamble_corr': np.max(postamble_correlation)
        }
        
        return extracted_signal, sync_info
    
    def simulate_transmission(self, padded_signal: np.ndarray, 
                            snr_db: float = 15.0,
                            add_before: int = 0,
                            add_after: int = 0,
                            cut_from_start: int = 0,
                            cut_from_end: int = 0) -> Tuple[np.ndarray, dict]:
        """
        Simulate realistic transmission with various impairments.
        
        Parameters:
        -----------
        padded_signal : np.ndarray
            The padded signal (with preamble and postamble)
        snr_db : float
            Signal-to-Noise Ratio in dB (default: 15.0)
        add_before : int
            Number of random noise samples to add before the signal (default: 0)
        add_after : int
            Number of random noise samples to add after the signal (default: 0)
        cut_from_start : int
            Number of samples to cut from the beginning (default: 0)
            This will remove part of the preamble
        cut_from_end : int
            Number of samples to cut from the end (default: 0)
            This will remove part of the postamble
            
        Returns:
        --------
        Tuple[np.ndarray, dict]
            - The transmitted signal after all impairments
            - Dictionary with transmission information
        """
        # Start with the padded signal
        transmitted = padded_signal.copy()
        
        # 1. Add AWGN noise
        signal_power = np.mean(transmitted ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(transmitted))
        transmitted = transmitted + noise
        
        # 2. Cut from start (simulate late recording start or preamble damage)
        if cut_from_start > 0:
            cut_from_start = min(cut_from_start, len(transmitted))
            transmitted = transmitted[cut_from_start:]
        
        # 3. Cut from end (simulate early recording stop or postamble damage)
        if cut_from_end > 0:
            cut_from_end = min(cut_from_end, len(transmitted))
            transmitted = transmitted[:-cut_from_end]
        
        # 4. Add random noise before (simulate early recording start)
        if add_before > 0:
            noise_before = np.sqrt(noise_power) * np.random.randn(add_before)
            transmitted = np.concatenate([noise_before, transmitted])
        
        # 5. Add random noise after (simulate late recording stop)
        if add_after > 0:
            noise_after = np.sqrt(noise_power) * np.random.randn(add_after)
            transmitted = np.concatenate([transmitted, noise_after])
        
        # Create info dictionary
        transmission_info = {
            'snr_db': snr_db,
            'noise_power': noise_power,
            'signal_power': signal_power,
            'add_before': add_before,
            'add_after': add_after,
            'cut_from_start': cut_from_start,
            'cut_from_end': cut_from_end,
            'original_length': len(padded_signal),
            'transmitted_length': len(transmitted),
            'length_change': len(transmitted) - len(padded_signal)
        }
        
        return transmitted, transmission_info
