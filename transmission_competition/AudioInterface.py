"""
File: AudioInterface.py
Author: Felix Egger
Description: Implements an audio interface for reading, writing, and playing audio files.
"""
import numpy as np
from scipy.io import wavfile
import os

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class AudioInterface:
    def __init__(self):
        """Initialize the AudioInterface."""
        pass

    def write_audio(self, signal: np.ndarray, sample_rate: int, file_path: str) -> None:
        # Normalize signal to [-1, 1]
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            normalized = signal / max_val
        else:
            normalized = signal

        # Convert to 16-bit integer format
        audio_data = (normalized * 32767).astype(np.int16)

        # Write to WAV file
        wavfile.write(file_path, sample_rate, audio_data)

    def read_audio(self, file_path: str) -> tuple[np.ndarray, int]:
        sample_rate, audio_data = wavfile.read(file_path)

        # Convert to float and normalize to [-1, 1]
        if audio_data.dtype == np.int16:
            signal = audio_data.astype(np.float64) / 32767.0
        elif audio_data.dtype == np.int32:
            signal = audio_data.astype(np.float64) / 2147483647.0
        elif audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            signal = audio_data.astype(np.float64)
        else:
            signal = audio_data.astype(np.float64)

        return signal, sample_rate

    def play_audio(self, file_path: str) -> None:
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice not available.")

        sample_rate, audio_data = wavfile.read(file_path)
        sd.play(audio_data, sample_rate)
        sd.wait()

    def play_signal(self, signal: np.ndarray, sample_rate: int) -> None:
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice not available.")

        # Normalize signal to [-1, 1]
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            normalized = signal / max_val
        else:
            normalized = signal

        sd.play(normalized, sample_rate)
        sd.wait()