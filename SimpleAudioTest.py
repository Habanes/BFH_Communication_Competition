"""
Simple Audio Transmission Test - Just Synchronization
Test sending a simple bit sequence with preamble/postamble
"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from transmission_competition.Synchroniser import Synchroniser
from transmission_competition.CSSModulator import CSSModulator


class SimpleAudioTest:
    def __init__(self):
        self.sample_rate = 48000.0
        self.symbol_rate = 4.0  # symbols per second
        
        # Modulator for converting bits to audio
        T_symbol = 1.0 / self.symbol_rate
        self.modulator = CSSModulator(fs=self.sample_rate, T_symbol=T_symbol, 
                                     f_start=1000.0, bandwidth=3000.0)
        
        # Synchroniser with default parameters (matching notebook)
        self.synchroniser = Synchroniser(preamble_length=1000, postamble_length=1000, 
                                        f0=100, f1=20000, sample_rate=self.sample_rate)
    
    def send_test_pattern(self, bits: np.ndarray):
        """Send a test bit pattern over audio."""
        print(f"\n{'='*70}")
        print("SENDER MODE")
        print(f"{'='*70}")
        print(f"Test bits: {bits}")
        print(f"Number of bits: {len(bits)}")
        
        # Modulate bits to audio signal
        modulated = self.modulator.CSS_modulate(bits)
        print(f"Modulated signal: {len(modulated)} samples ({len(modulated)/self.sample_rate:.2f}s)")
        
        # Add preamble and postamble
        padded = self.synchroniser.pad(modulated)
        print(f"Padded signal: {len(padded)} samples ({len(padded)/self.sample_rate:.2f}s)")
        print(f"  - Preamble: {self.synchroniser.preamble_length} samples")
        print(f"  - Data: {len(modulated)} samples")
        print(f"  - Postamble: {self.synchroniser.postamble_length} samples")
        
        # Normalize for playback
        signal = padded / np.max(np.abs(padded)) * 0.8
        
        print(f"\nRecommended recording duration: {len(signal)/self.sample_rate + 2:.0f} seconds")
        print(f"\nPress 'y' + ENTER to transmit: ", end='')
        if input().strip().lower() != 'y':
            print("Cancelled.")
            return
        
        print("🔊 Transmitting...")
        sd.play(signal, samplerate=self.sample_rate, blocking=True)
        print("✓ Transmission complete!\n")
    
    def receive_test_pattern(self, duration: float = 10.0):
        """Record and decode a test pattern."""
        print(f"\n{'='*70}")
        print("RECEIVER MODE")
        print(f"{'='*70}")
        print(f"Recording for {duration:.1f} seconds...")
        print("Press ENTER to start recording: ", end='')
        input()
        
        # Record audio
        n_samples = int(duration * self.sample_rate)
        print("🎤 Recording...")
        recording = sd.rec(n_samples, samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()
        signal = recording.flatten()
        print(f"✓ Recorded {len(signal)} samples ({len(signal)/self.sample_rate:.1f}s)")
        
        # Signal statistics
        max_amp = np.max(np.abs(signal))
        print(f"\nSignal statistics:")
        print(f"  Max amplitude: {max_amp:.4f}")
        print(f"  RMS: {np.sqrt(np.mean(signal**2)):.4f}")
        
        if max_amp < 0.01:
            print("  ⚠️  WARNING: Signal is very quiet!")
        
        # Visualize option
        viz = input("\nVisualize signal? (y/n): ").strip().lower()
        if viz == 'y':
            self.visualize(signal)
        
        # Remove preamble/postamble
        print("\nExtracting signal...")
        extracted, sync_info = self.synchroniser.depad(signal)
        
        print(f"\nSynchronization info:")
        print(f"  Preamble correlation: {sync_info['max_preamble_corr']:.4f}")
        print(f"  Postamble correlation: {sync_info['max_postamble_corr']:.4f}")
        print(f"  Preamble detected at: sample {sync_info['preamble_idx']}")
        print(f"  Postamble detected at: sample {sync_info['postamble_idx']}")
        print(f"  Signal start: {sync_info['signal_start']}")
        print(f"  Signal end: {sync_info['signal_end']}")
        print(f"  Extracted length: {len(extracted)} samples")
        
        if sync_info['max_preamble_corr'] < 0.1:
            print("  ⚠️  WARNING: Preamble detection is very weak!")
        
        if len(extracted) == 0:
            print("✗ No signal extracted - synchronization failed!")
            return None
        
        # Demodulate
        print("\nDemodulating...")
        bits = self.modulator.CSS_demodulate(extracted)
        print(f"Demodulated bits: {bits}")
        print(f"Number of bits: {len(bits)}")
        
        return bits
    
    def visualize(self, signal: np.ndarray):
        """Visualize the recorded signal."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Full signal
        time = np.arange(len(signal)) / self.sample_rate
        axes[0].plot(time, signal, linewidth=0.5, alpha=0.7)
        axes[0].set_title(f'Full Signal ({len(signal)/self.sample_rate:.2f}s)', fontweight='bold')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Zoomed (first 0.5 seconds)
        N_zoom = min(int(0.5 * self.sample_rate), len(signal))
        time_zoom = np.arange(N_zoom) / self.sample_rate
        axes[1].plot(time_zoom, signal[:N_zoom], linewidth=1.0)
        axes[1].set_title('First 0.5 Seconds', fontweight='bold')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        # Frequency spectrum
        N_fft = min(8192, len(signal))
        fft_result = np.fft.fft(signal[:N_fft])
        freqs = np.fft.fftfreq(N_fft, 1/self.sample_rate)
        magnitude = np.abs(fft_result)
        
        mask = (freqs >= 0) & (freqs <= 5000)
        axes[2].plot(freqs[mask], magnitude[mask], linewidth=1.0)
        axes[2].set_title('Frequency Spectrum (0-5000 Hz)', fontweight='bold')
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('Magnitude')
        axes[2].grid(True, alpha=0.3)
        
        # Mark expected frequencies
        axes[2].axvline(x=self.synchroniser.f0, color='blue', linestyle='--', 
                       alpha=0.5, label=f'f0: {self.synchroniser.f0:.0f}Hz')
        axes[2].axvline(x=self.synchroniser.f1, color='red', linestyle='--', 
                       alpha=0.5, label=f'f1: {self.synchroniser.f1:.0f}Hz')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Preamble detection test
        preamble_norm = self.synchroniser.preamble / np.linalg.norm(self.synchroniser.preamble)
        correlation = np.correlate(signal, preamble_norm, mode='valid')
        max_idx = np.argmax(np.abs(correlation))
        max_val = np.abs(correlation[max_idx])
        
        print(f"\nManual preamble detection:")
        print(f"  Max correlation: {max_val:.4f} at sample {max_idx} ({max_idx/self.sample_rate:.2f}s)")


def main():
    tester = SimpleAudioTest()
    
    print("\n" + "="*70)
    print("  SIMPLE AUDIO SYNCHRONIZATION TEST")
    print("="*70)
    print("\n[1] Send test pattern (20 bits)")
    print("[2] Receive and decode")
    print("[0] Exit\n")
    
    choice = input("Select mode: ").strip()
    
    if choice == '1':
        # Create a test pattern
        test_bits = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1])
        tester.send_test_pattern(test_bits)
        
        print(f"\n{'='*70}")
        print("REFERENCE PATTERN FOR RECEIVER:")
        print(f"Expected bits: {test_bits}")
        print(f"{'='*70}\n")
    
    elif choice == '2':
        duration_input = input("\nRecording duration (seconds, default=10): ").strip()
        duration = float(duration_input) if duration_input else 10.0
        
        received_bits = tester.receive_test_pattern(duration)
        
        if received_bits is not None:
            print(f"\n{'='*70}")
            print("RESULT:")
            print(f"Received bits: {received_bits}")
            print(f"{'='*70}")
            
            # Compare with expected
            expected = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1])
            if len(received_bits) == len(expected):
                if np.array_equal(received_bits, expected):
                    print("✓ SUCCESS! Bits match perfectly!")
                else:
                    errors = np.sum(received_bits != expected)
                    print(f"✗ MISMATCH: {errors} bit errors out of {len(expected)}")
            else:
                print(f"✗ LENGTH MISMATCH: Expected {len(expected)} bits, got {len(received_bits)}")
    
    else:
        print("Goodbye!")


if __name__ == "__main__":
    main()
