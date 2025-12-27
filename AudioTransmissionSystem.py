"""
Audio Transmission System - Send/Receive text over audio
"""

import numpy as np
try:
    import sounddevice as sd
except ImportError:
    print("Warning: sounddevice not available")
    sd = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not available")
    plt = None

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
        self.symbol_rate = 4.0  # 20 symbols per second
        
        # Initialize components
        self.source_coder = HuffmanCoder()
        self.channel_coder = HammingCoder74()
        
        # CSS Modulator with proper symbol duration
        T_symbol = 1.0 / self.symbol_rate  # Symbol duration in seconds
        self.modulator = CSSModulator(fs=self.sample_rate, T_symbol=T_symbol, 
                                     f_start=1000.0, bandwidth=3000.0)
        
        # Synchroniser with chirp frequencies matching notebook defaults
        # Using wider frequency range (100-20000 Hz) for better detection
        self.synchroniser = Synchroniser(preamble_length=1000, postamble_length=1000, 
                                        f0=100, f1=20000, sample_rate=self.sample_rate)
        
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
        num_bits = len(channel_coded)
        print(f"✓ Signal ready: {num_bits} bits, {duration:.2f}s duration\n")
        print(f"   Tip: Set receiver duration to at least {duration + 2:.0f} seconds\n")
        return signal
    
    def receive(self, signal: np.ndarray) -> str:
        print("\n📥 Decoding received signal...")
        print(f"   Received {len(signal)} samples ({len(signal)/self.sample_rate:.1f}s)")
        
        # Remove sync and demodulate
        extracted, sync_info = self.synchroniser.depad(signal)
        
        # Check sync quality
        preamble_corr = sync_info['max_preamble_corr']
        postamble_corr = sync_info['max_postamble_corr']
        print(f"   Preamble sync: {preamble_corr:.3f}")
        print(f"   Postamble sync: {postamble_corr:.3f}")
        
        if preamble_corr < 0.1:
            print("   ⚠️  WARNING: Very weak preamble detection!")
            print("   Possible issues: No signal received, too much noise, or wrong parameters")
        
        if postamble_corr < 0.1:
            print("   ⚠️  WARNING: Very weak postamble detection!")
            print("   This might cause incorrect signal extraction")
        
        if len(extracted) == 0:
            return "[Error: No signal extracted - synchronization failed]"
        
        print(f"   Extracted {len(extracted)} samples")
        
        demodulated = self.modulator.CSS_demodulate(extracted)
        print(f"   Demodulated {len(demodulated)} bits")
        
        if len(demodulated) == 0:
            return "[Error: No bits demodulated]"
        
        # For debugging: show first few bits
        print(f"   First 20 bits: {demodulated[:20]}")
        
        # Trim to multiple of 7 for Hamming
        original_length = len(demodulated)
        if len(demodulated) % 7 != 0:
            demodulated = demodulated[:(len(demodulated) // 7) * 7]
            print(f"   Trimmed from {original_length} to {len(demodulated)} bits (multiple of 7)")
        
        if len(demodulated) == 0:
            return "[Error: Not enough bits for channel decoding]"
        
        # Decode
        try:
            channel_decoded = self.channel_coder.decode(demodulated)
            print(f"   Channel decoded {len(channel_decoded)} bits")
            
            if len(channel_decoded) == 0:
                return "[Error: Channel decoding produced no bits]"
            
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
    
    def visualize(self, signal: np.ndarray, title: str = "Signal Analysis"):
        """Visualize the received signal to help debug issues."""
        if plt is None:
            print("Matplotlib not available for visualization")
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Time domain - full signal
        time = np.arange(len(signal)) / self.sample_rate
        axes[0].plot(time, signal, linewidth=0.5, alpha=0.7)
        axes[0].set_title(f'{title} - Full Signal ({len(signal)/self.sample_rate:.2f}s)', fontweight='bold')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Time domain - zoomed in (first 2 seconds)
        N_zoom = min(int(2 * self.sample_rate), len(signal))
        time_zoom = np.arange(N_zoom) / self.sample_rate
        axes[1].plot(time_zoom, signal[:N_zoom], linewidth=1.0)
        axes[1].set_title('First 2 Seconds (zoomed)', fontweight='bold')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        # Frequency domain (FFT)
        N_fft = min(8192, len(signal))
        fft_result = np.fft.fft(signal[:N_fft])
        freqs = np.fft.fftfreq(N_fft, 1/self.sample_rate)
        magnitude = np.abs(fft_result)
        
        # Only plot positive frequencies up to 5000 Hz
        mask = (freqs >= 0) & (freqs <= 5000)
        axes[2].plot(freqs[mask], magnitude[mask], linewidth=1.0)
        axes[2].set_title('Frequency Spectrum (0-5000 Hz)', fontweight='bold')
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('Magnitude')
        axes[2].grid(True, alpha=0.3)
        
        # Add markers for expected frequencies
        axes[2].axvline(x=self.synchroniser.f0, color='blue', linestyle='--', 
                       alpha=0.5, label=f'Preamble start: {self.synchroniser.f0:.0f}Hz')
        axes[2].axvline(x=self.synchroniser.f1, color='red', linestyle='--', 
                       alpha=0.5, label=f'Preamble end: {self.synchroniser.f1:.0f}Hz')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print signal statistics
        print("\n📊 Signal Statistics:")
        print(f"   Length: {len(signal)} samples ({len(signal)/self.sample_rate:.2f}s)")
        print(f"   Max amplitude: {np.max(np.abs(signal)):.4f}")
        print(f"   RMS: {np.sqrt(np.mean(signal**2)):.4f}")
        print(f"   Peak-to-peak: {np.max(signal) - np.min(signal):.4f}")
        
        # Check if signal is too quiet
        if np.max(np.abs(signal)) < 0.01:
            print("   ⚠️  Signal is very quiet - check microphone volume!")
        
        # Try to detect preamble manually
        preamble = self.synchroniser.preamble
        preamble_norm = preamble / np.linalg.norm(preamble)
        correlation = np.correlate(signal, preamble_norm, mode='valid')
        max_corr_idx = np.argmax(np.abs(correlation))
        max_corr_val = np.abs(correlation[max_corr_idx])
        
        print(f"\n🔍 Preamble Detection:")
        print(f"   Max correlation: {max_corr_val:.4f}")
        print(f"   Location: {max_corr_idx} samples ({max_corr_idx/self.sample_rate:.2f}s)")
        if max_corr_val > 0.1:
            print(f"   ✓ Preamble likely detected!")
        else:
            print(f"   ✗ Preamble NOT detected - signal may be missing or corrupted")
        print()
    
    def test_encoding(self, text: str):
        """Test the encoding/decoding pipeline without audio."""
        print(f"\n🧪 Testing encoding pipeline for: '{text}'")
        
        # Encode
        self.source_coder.build_encoding_map(text)
        source_coded = self.source_coder.encode(text)
        print(f"Source coded: {len(source_coded)} bits")
        
        channel_coded = self.channel_coder.encode(source_coded)
        print(f"Channel coded: {len(channel_coded)} bits")
        
        # Modulate
        modulated = self.modulator.CSS_modulate(channel_coded)
        print(f"Modulated: {len(modulated)} samples")
        
        # Add sync
        signal = self.synchroniser.pad(modulated)
        print(f"With sync: {len(signal)} samples")
        
        # Simulate reception (remove sync)
        extracted, sync_info = self.synchroniser.depad(signal)
        print(f"Extracted: {len(extracted)} samples")
        print(f"Sync quality - Preamble: {sync_info['max_preamble_corr']:.3f}, Postamble: {sync_info['max_postamble_corr']:.3f}")
        
        # Demodulate
        demodulated = self.modulator.CSS_demodulate(extracted)
        print(f"Demodulated: {len(demodulated)} bits")
        
        # Trim for Hamming
        if len(demodulated) % 7 != 0:
            demodulated = demodulated[:(len(demodulated) // 7) * 7]
        print(f"After trimming: {len(demodulated)} bits")
        
        # Decode
        channel_decoded = self.channel_coder.decode(demodulated)
        print(f"Channel decoded: {len(channel_decoded)} bits")
        
        decoded_text = self.source_coder.decode(channel_decoded)
        print(f"Final result: '{decoded_text}'")
        
        success = decoded_text == text
        print(f"✓ Success: {success}")
        return success
    
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
    print("[3] Test encoding pipeline")
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
        
        # Ask if user wants to visualize first
        viz = input("\nVisualize signal before decoding? (y/n): ").strip().lower()
        if viz == 'y':
            system.visualize(recording, "Received Signal")
        
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
    
    elif choice == '3':
        # TEST ENCODING
        print("\n--- TEST ENCODING PIPELINE ---")
        test_text = input("Enter test text (default='Hello World'): ").strip() or "Hello World"
        system.test_encoding(test_text)


if __name__ == "__main__":
    main()