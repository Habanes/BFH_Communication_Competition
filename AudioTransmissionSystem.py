"""
Audio Transmission System - Send/Receive text over audio
"""

import numpy as np
import sounddevice as sd
import os

from transmission_competition.HuffmanCoder import HuffmanCoder
from transmission_competition.HammingCoder74 import HammingCoder74
from transmission_competition.CSSModulator import CSSModulator
from transmission_competition.Synchroniser import Synchroniser


class AudioTransmissionSystem:
    def __init__(self):
        # Audio sample rate for playback (DO NOT CHANGE - standard audio rate)
        self.sample_rate = 48000.0
        
        # Symbol rate - controls transmission speed (lower = slower, more robust)
        self.symbol_rate = 20.0  # 20 symbols per second
        
        # Initialize components
        self.source_coder = HuffmanCoder()
        self.channel_coder = HammingCoder74()
        
        # CSS Modulator with proper symbol duration
        T_symbol = 1.0 / self.symbol_rate  # Symbol duration in seconds
        self.modulator = CSSModulator(fs=self.sample_rate, T_symbol=T_symbol, 
                                     f_start=1000.0, bandwidth=3000.0)
        
        # Synchroniser with chirp frequencies suitable for audio
        self.synchroniser = Synchroniser(preamble_length=4000, postamble_length=4000, 
                                        f0=500, f1=4000, sample_rate=self.sample_rate)
        
    def send(self, text: str) -> np.ndarray:
        print(f"\n📤 Preparing to send: '{text}'")
        
        # Encode
        self.source_coder.build_encoding_map(text)
        source_coded = self.source_coder.encode(text)
        channel_coded = self.channel_coder.encode(source_coded)
        
        # Modulate and add sync
        modulated = self.modulator.CSS_modulate(channel_coded)
        signal = self.synchroniser.pad(modulated)
        
        duration = len(signal) / self.sample_rate
        print(f"✓ Signal ready ({duration:.2f}s)\n")
        return signal
    
    def receive(self, signal: np.ndarray) -> str:
        print("\n📥 Decoding received signal...")
        
        # Remove sync and demodulate
        extracted, sync_info = self.synchroniser.depad(signal)
        print(f"   Sync quality: {sync_info['max_preamble_corr']:.2f}")
        
        demodulated = self.modulator.CSS_demodulate(extracted)
        
        # Trim to multiple of 7 for Hamming
        if len(demodulated) % 7 != 0:
            demodulated = demodulated[:(len(demodulated) // 7) * 7]
        
        # Decode
        try:
            channel_decoded = self.channel_coder.decode(demodulated)
            text = self.source_coder.decode(channel_decoded)
            print(f"✓ Decoded: '{text}'\n")
            return text
        except Exception as e:
            print(f"✗ Decoding failed: {e}\n")
            return f"[Error: {e}]"
    
    def play(self, signal: np.ndarray):
        print("Press 'y' + ENTER to transmit: ", end='')
        if input().strip().lower() != 'y':
            print("Cancelled.")
            return
        
        # Normalize and play
        signal = signal / np.max(np.abs(signal)) * 0.8
        print("🔊 Transmitting...")
        sd.play(signal, samplerate=self.sample_rate, blocking=True)
        print("✓ Complete\n")
    
    def record(self, duration: float = 10.0) -> np.ndarray:
        print(f"🎤 Recording for {duration:.1f}s...")
        print("Press ENTER to start: ", end='')
        input()
        
        n_samples = int(duration * self.sample_rate)
        recording = sd.rec(n_samples, samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()
        
        print(f"✓ Recorded {len(recording.flatten())/self.sample_rate:.1f}s\n")
        return recording.flatten()


def main():
    system = AudioTransmissionSystem()
    
    print("\n" + "="*70)
    print("  AUDIO TRANSMISSION SYSTEM")
    print("="*70)
    print("\n[1] Send message")
    print("[2] Receive message")
    print("[0] Exit\n")
    
    choice = input("Select mode: ").strip()
    
    if choice == '1':
        # SENDER
        print("\n--- SENDER MODE ---")
        print("\nEnter message (or file path starting with 'file:'):")
        user_input = input().strip()
        
        if user_input.startswith('file:'):
            filepath = user_input[5:].strip()
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                print(f"Loaded from {filepath}")
            else:
                print(f"File not found: {filepath}")
                return
        else:
            text = user_input
        
        signal = system.send(text)
        system.play(signal)
    
    elif choice == '2':
        # RECEIVER
        print("\n--- RECEIVER MODE ---")
        duration_input = input("\nRecording duration (seconds, default=10): ").strip()
        duration = float(duration_input) if duration_input else 10.0
        
        recording = system.record(duration)
        text = system.receive(recording)
        
        print("="*70)
        print(f"RECEIVED: {text}")
        print("="*70)
        
        save = input("\nSave to file? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("Filename (default=received.txt): ").strip() or "received.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"✓ Saved to {filename}")
    
    else:
        print("Goodbye!")


if __name__ == "__main__":
    main()