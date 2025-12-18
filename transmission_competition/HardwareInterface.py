"""
File: HardwareInterface.py
Author: AI Assistant
Description: Hardware interface for audio transmission using laptop microphone/speaker
"""

import numpy as np
import sounddevice as sd
from typing import Optional, Tuple
import time


class HardwareInterface:
    """
    Hardware interface for transmitting and receiving modulated signals via audio.
    Uses laptop speaker for transmission and microphone for reception.
    Supports chirp modulation with preamble synchronization.
    """
    
    def __init__(self, fs: float = 48000.0, preamble_duration: float = 0.1, 
                 preamble_freq: float = 5000.0, silence_duration: float = 0.05):
        """
        Initialize hardware interface parameters.
        
        :param fs: Sampling frequency [Hz] (default: 48000.0)
        :param preamble_duration: Duration of preamble signal [s] (default: 0.1)
        :param preamble_freq: Frequency of preamble tone [Hz] (default: 5000.0)
        :param silence_duration: Silence duration after preamble [s] (default: 0.05)
        """
        self.fs = fs
        self.preamble_duration = preamble_duration
        self.preamble_freq = preamble_freq
        self.silence_duration = silence_duration
        
        # Generate preamble signal (sine wave)
        n_samples_preamble = int(self.preamble_duration * self.fs)
        t_preamble = np.linspace(0, self.preamble_duration, n_samples_preamble, endpoint=False)
        self.preamble_signal = np.sin(2 * np.pi * self.preamble_freq * t_preamble)
        
        # Add silence after preamble
        n_samples_silence = int(self.silence_duration * self.fs)
        self.silence = np.zeros(n_samples_silence)
        
        # Recording parameters
        self.recorded_signal: Optional[np.ndarray] = None
        self.is_recording = False
        
    def generate_transmission_signal(self, modulated_signal: np.ndarray) -> np.ndarray:
        """
        Prepare signal for transmission by adding preamble and silence.
        
        :param modulated_signal: Modulated signal array to transmit
        :return: Complete transmission signal with preamble
        """
        # Concatenate: preamble -> silence -> modulated signal
        tx_signal = np.concatenate([self.preamble_signal, self.silence, modulated_signal])
        return tx_signal
    
    def transmit(self, modulated_signal: np.ndarray, blocking: bool = True):
        """
        Transmit modulated signal through speaker.
        
        :param modulated_signal: Modulated signal array to transmit
        :param blocking: If True, wait for playback to complete (default: True)
        """
        tx_signal = self.generate_transmission_signal(modulated_signal)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(tx_signal))
        if max_val > 0:
            tx_signal = tx_signal / max_val * 0.9  # 0.9 to leave some headroom
        
        print(f"Transmitting signal: {len(tx_signal)} samples ({len(tx_signal)/self.fs:.3f} seconds)")
        sd.play(tx_signal, samplerate=self.fs, blocking=blocking)
        
        if not blocking:
            print("Transmission started in background. Use sd.wait() to wait for completion.")
    
    def start_transmission(self, modulated_signal: np.ndarray):
        """
        Start transmitting the modulated signal with preamble.
        Convenience method that calls transmit() with blocking=True.
        
        :param modulated_signal: Modulated signal array to transmit
        """
        self.transmit(modulated_signal, blocking=True)
        print("Transmission complete.")
    
    def record(self, duration: float) -> np.ndarray:
        """
        Record audio from microphone for specified duration.
        
        :param duration: Recording duration [s]
        :return: Recorded audio signal as numpy array
        """
        n_samples = int(duration * self.fs)
        print(f"Recording for {duration:.3f} seconds...")
        
        recording = sd.rec(n_samples, samplerate=self.fs, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        
        self.recorded_signal = recording.flatten()
        print(f"Recording complete: {len(self.recorded_signal)} samples")
        return self.recorded_signal
    
    def detect_preamble(self, signal: np.ndarray, threshold: float = 0.5) -> Optional[int]:
        """
        Detect preamble in recorded signal using cross-correlation.
        
        :param signal: Recorded signal to search for preamble
        :param threshold: Correlation threshold for detection (0-1, default: 0.5)
        :return: Index where data starts (after preamble and silence), or None if not found
        """
        # Cross-correlate signal with preamble
        correlation = np.correlate(signal, self.preamble_signal, mode='valid')
        
        # Normalize correlation
        norm_factor = np.sqrt(np.sum(self.preamble_signal**2))
        if norm_factor > 0:
            correlation = correlation / norm_factor
        
        # Find maximum correlation
        max_corr_idx = np.argmax(np.abs(correlation))
        max_corr_val = np.abs(correlation[max_corr_idx])
        
        print(f"Preamble detection: max correlation = {max_corr_val:.3f} at index {max_corr_idx}")
        
        if max_corr_val >= threshold:
            # Data starts after preamble and silence
            data_start_idx = max_corr_idx + len(self.preamble_signal) + len(self.silence)
            print(f"Preamble detected! Data starts at index {data_start_idx}")
            return data_start_idx
        else:
            print(f"Preamble not detected (threshold: {threshold})")
            return None
    
    def receive(self, duration: float, auto_detect_preamble: bool = True) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Record and extract the modulated signal from audio input.
        
        :param duration: Recording duration [s]
        :param auto_detect_preamble: Automatically detect preamble and extract data (default: True)
        :return: Tuple of (data_signal, data_start_index). If preamble not detected, returns (None, None)
        """
        recorded = self.record(duration)
        
        if not auto_detect_preamble:
            return recorded, 0
        
        data_start = self.detect_preamble(recorded)
        
        if data_start is not None and data_start < len(recorded):
            data_signal = recorded[data_start:]
            return data_signal, data_start
        else:
            print("Warning: Could not extract data signal from recording")
            return None, None
    
    def start_reception(self, duration: float) -> Optional[np.ndarray]:
        """
        Start receiving and return the extracted data signal.
        Convenience method that automatically detects preamble.
        
        :param duration: Expected reception duration [s]
        :return: Extracted data signal, or None if preamble not detected
        """
        data_signal, _ = self.receive(duration, auto_detect_preamble=True)
        return data_signal
    
    def get_device_info(self):
        """
        Print information about available audio devices.
        """
        print("\n" + "="*70)
        print("Audio Device Information:")
        print("="*70)
        print(sd.query_devices())
        print("="*70 + "\n")
    
    def stop(self):
        """
        Stop any ongoing playback or recording.
        """
        sd.stop()
        print("Audio stopped.")


if __name__ == "__main__":
    # Test hardware interface
    print("Testing HardwareInterface")
    print("="*70)
    
    hw = HardwareInterface()
    hw.get_device_info()
    
    # Generate a test signal (simple chirp)
    from CSSModulation import CSSModulator
    
    css = CSSModulator()
    test_bits = np.array([0, 1, 0, 1, 1, 0])
    test_signal = css.CSS_modulate(test_bits)
    
    print(f"\nTest signal: {len(test_signal)} samples ({len(test_signal)/css.fs:.3f} seconds)")
    print(f"Test bits: {test_bits}")
    
    # Test transmission
    print("\n--- Testing Transmission ---")
    input("Press Enter to start transmission...")
    hw.start_transmission(test_signal)
    
    # Test reception
    print("\n--- Testing Reception ---")
    reception_duration = (len(test_signal) + len(hw.preamble_signal) + len(hw.silence)) / hw.fs + 0.5
    print(f"Recording for {reception_duration:.2f} seconds (with buffer)")
    input("Press Enter to start recording...")
    
    received_data = hw.start_reception(reception_duration)
    
    if received_data is not None:
        print(f"\nReceived data signal: {len(received_data)} samples")
        
        # Try to demodulate
        demod_bits = css.CSS_demodulate(received_data[:len(test_signal)])
        print(f"Demodulated bits: {demod_bits[:len(test_bits)]}")
        print(f"Original bits:    {test_bits}")
        
        errors = np.sum(test_bits != demod_bits[:len(test_bits)])
        print(f"Bit errors: {errors}/{len(test_bits)}")
    else:
        print("\nReception failed - preamble not detected")
