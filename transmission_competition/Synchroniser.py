import numpy as np
from typing import Tuple


class Synchroniser:
    """
    Synchronisation class for adding and removing padding to/from signals.
    Uses chirp signals (frequency ramps) as preamble and postamble for synchronization.
    """
    
    def __init__(self, preamble_length: int = 8820, postamble_length: int = 8820,
                 f0: float = 200, f1: float = 18000, sample_rate: float = 44100):
        """
        Initialize the Synchronisation class with best-practice chirp patterns.
        
        Parameters:
        -----------
        preamble_length : int
            Length of the preamble in samples (default: 8820, ~200ms at 44.1kHz)
            Longer chirps provide better correlation peaks for reliable detection
        postamble_length : int
            Length of the postamble in samples (default: 8820, ~200ms at 44.1kHz)
        f0 : float
            Starting frequency for chirp in Hz (default: 200)
            Set above 100 Hz to avoid low-frequency noise
        f1 : float
            Ending frequency for chirp in Hz (default: 18000)
            Below Nyquist frequency (22.05 kHz) with margin
        sample_rate : float
            Sample rate in Hz (default: 44100)
        """
        self.preamble_length = preamble_length
        self.postamble_length = postamble_length
        self.f0 = f0
        self.f1 = f1
        self.sample_rate = sample_rate
        
        # Generate the preamble and postamble chirps with optimal parameters
        self.preamble = self._generate_chirp(self.preamble_length, self.f0, self.f1)
        self.postamble = self._generate_chirp(self.postamble_length, self.f1, self.f0)  # Down-chirp
        
        # Pre-compute normalized matched filter templates for correlation
        self.preamble_template = self.preamble / np.sqrt(np.sum(self.preamble ** 2))
        self.postamble_template = self.postamble / np.sqrt(np.sum(self.postamble ** 2))
    
    def _generate_chirp(self, length: int, f_start: float, f_end: float) -> np.ndarray:
        """
        Generate an optimal linear chirp signal for matched filter detection.
        
        Best practices implemented:
        - Linear frequency sweep for predictable autocorrelation properties
        - Tukey window (tapered cosine) for smooth transitions while maintaining energy
        - High time-bandwidth product for robust detection in noise
        
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
            The generated chirp signal optimized for cross-correlation
        """
        t = np.arange(length) / self.sample_rate
        # Linear chirp: instantaneous frequency f(t) = f_start + k*t
        chirp_duration = length / self.sample_rate
        k = (f_end - f_start) / chirp_duration  # Chirp rate (Hz/s)
        
        # Phase: integral of 2*pi*f(t) = 2*pi*(f_start*t + 0.5*k*t^2)
        phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
        chirp = np.sin(phase)
        
        # Apply Tukey window (better than Hanning for correlation)
        # Tukey window: flat top with cosine tapers at edges (alpha=0.1 means 10% taper)
        # This preserves more signal energy than Hanning while still reducing edge effects
        alpha = 0.1  # Taper fraction
        window = self._tukey_window(length, alpha)
        chirp = chirp * window
        
        # Normalize to unit energy for optimal matched filtering
        chirp = chirp / np.sqrt(np.sum(chirp ** 2))
        
        return chirp
    
    def _tukey_window(self, length: int, alpha: float = 0.1) -> np.ndarray:
        """
        Generate a Tukey window (tapered cosine window).
        
        Parameters:
        -----------
        length : int
            Length of the window
        alpha : float
            Taper fraction (0 = rectangular, 1 = Hann window)
            
        Returns:
        --------
        np.ndarray
            The Tukey window
        """
        n = np.arange(length)
        window = np.ones(length)
        
        # Taper at the beginning
        taper_len = int(alpha * length / 2)
        if taper_len > 0:
            window[:taper_len] = 0.5 * (1 + np.cos(np.pi * (2 * n[:taper_len] / (alpha * length) - 1)))
            # Taper at the end
            window[-taper_len:] = 0.5 * (1 + np.cos(np.pi * (2 * (n[-taper_len:] - (length - taper_len)) / (alpha * length) + 1)))
        
        return window
    
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
    
    def depad(self, received_signal: np.ndarray, correlation_threshold: float = 0.3) -> Tuple[np.ndarray, dict]:
        """
        Remove preamble and postamble from received signal using matched filter detection.
        
        Uses cross-correlation (matched filtering) with normalized templates for
        optimal detection of chirp patterns in noise. This is the best practice
        approach for synchronization in digital communications.
        
        Parameters:
        -----------
        received_signal : np.ndarray
            The received signal with preamble and postamble
        correlation_threshold : float
            Minimum normalized correlation value for valid detection (0 to 1, default: 0.3)
            Lower values are more permissive but may cause false detections
            
        Returns:
        --------
        Tuple[np.ndarray, dict]
            - The extracted signal without padding
            - Dictionary with detailed synchronization information
        """
        # Matched filter detection: cross-correlate with known templates
        # This maximizes SNR and provides optimal detection in AWGN
        
        # Correlate with preamble template (reversed for cross-correlation)
        preamble_correlation = np.correlate(received_signal, self.preamble_template[::-1], mode='valid')
        
        # Correlate with postamble template
        postamble_correlation = np.correlate(received_signal, self.postamble_template[::-1], mode='valid')
        
        # Find peaks in correlation (use absolute value for robustness to phase shifts)
        preamble_corr_abs = np.abs(preamble_correlation)
        postamble_corr_abs = np.abs(postamble_correlation)
        
        # Locate maximum correlation indices
        preamble_idx = np.argmax(preamble_corr_abs)
        postamble_idx = np.argmax(postamble_corr_abs)
        
        # Get correlation peak values (normalized)
        max_preamble_corr = preamble_corr_abs[preamble_idx]
        max_postamble_corr = postamble_corr_abs[postamble_idx]
        
        # The actual signal starts after the preamble ends
        # preamble_idx is where preamble starts, so signal starts at preamble_idx + preamble_length
        signal_start = preamble_idx + self.preamble_length
        signal_end = postamble_idx
        
        # Validation checks
        detection_valid = True
        warnings = []
        
        if max_preamble_corr < correlation_threshold:
            warnings.append(f"Weak preamble correlation: {max_preamble_corr:.4f} < {correlation_threshold}")
            detection_valid = False
            
        if max_postamble_corr < correlation_threshold:
            warnings.append(f"Weak postamble correlation: {max_postamble_corr:.4f} < {correlation_threshold}")
            detection_valid = False
        
        if signal_start >= signal_end:
            warnings.append(f"Invalid signal boundaries: start={signal_start} >= end={signal_end}")
            detection_valid = False
        
        # Extract the signal
        if detection_valid and signal_start >= 0 and signal_end <= len(received_signal):
            extracted_signal = received_signal[signal_start:signal_end]
        else:
            # Fallback: return entire signal with warnings
            extracted_signal = received_signal
            warnings.append("Synchronization failed - returning full received signal")
            if warnings:
                print("WARNING: " + "; ".join(warnings))
        
        # Compile detailed synchronization information
        sync_info = {
            'preamble_idx': int(preamble_idx),
            'postamble_idx': int(postamble_idx),
            'signal_start': int(signal_start),
            'signal_end': int(signal_end),
            'max_preamble_corr': float(max_preamble_corr),
            'max_postamble_corr': float(max_postamble_corr),
            'preamble_correlation': float(preamble_correlation[preamble_idx]),
            'postamble_correlation': float(postamble_correlation[postamble_idx]),
            'detection_valid': detection_valid,
            'warnings': warnings,
            'extracted_length': len(extracted_signal),
            'correlation_threshold': correlation_threshold
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
