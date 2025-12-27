"""
Simple Audio Transmission Test - Just Synchronization
Test sending exactly 20 bits with preamble detection
"""

import numpy as np
import sounddevice as sd
from transmission_competition.CSSModulator import CSSModulator


class SimpleAudioTest:
    def __init__(self):
        self.sample_rate = 48000.0
        self.symbol_rate = 4.0  # symbols per second
        self.n_bits = 20  # Fixed number of bits to send
        
        # Modulator for converting bits to audio
        T_symbol = 1.0 / self.symbol_rate
        self.modulator = CSSModulator(fs=self.sample_rate, T_symbol=T_symbol, 
                                     f_start=1000.0, bandwidth=3000.0)
        
        # Preamble: chirp from 100 Hz to 20 kHz
        self.preamble_length = 1000
        self.f0 = 100.0
        self.f1 = 20000.0
        self.preamble = self._generate_chirp()
    
    def _generate_chirp(self):
        """Generate a linear chirp signal for preamble."""
        t = np.arange(self.preamble_length) / self.sample_rate
        k = (self.f1 - self.f0) / (self.preamble_length / self.sample_rate)
        phase = 2 * np.pi * (self.f0 * t + 0.5 * k * t**2)
        chirp = np.sin(phase)
        # Apply window
        window = np.hanning(self.preamble_length)
        return chirp * window
    
    def send(self, bits: np.ndarray):
        """Send 20 bits over audio with preamble."""
        assert len(bits) == self.n_bits, f"Must send exactly {self.n_bits} bits!"
        
        print(f"Sending bits: {bits}")
        
        # Modulate bits to audio
        modulated = self.modulator.CSS_modulate(bits)
        
        # Add preamble
        signal = np.concatenate([self.preamble, modulated])
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        duration = len(signal) / self.sample_rate
        print(f"Signal: {len(signal)} samples ({duration:.2f}s)")
        print(f"Record for at least {duration + 1:.0f} seconds")
        
        # Transmit
        print("Transmitting...")
        sd.play(signal, samplerate=self.sample_rate, blocking=True)
        print("Done!\n")
    
    def receive(self, duration: float = 10.0):
        """Record and decode exactly 20 bits."""
        print(f"Recording for {duration:.1f}s...")
        input("Press ENTER to start: ")
        
        # Record
        n_samples = int(duration * self.sample_rate)
        recording = sd.rec(n_samples, samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()
        signal = recording.flatten()
        print(f"Recorded {len(signal)} samples")
        
        # Find preamble using autocorrelation
        preamble_norm = self.preamble / np.linalg.norm(self.preamble)
        correlation = np.correlate(signal, preamble_norm, mode='valid')
        correlation = np.abs(correlation)
        
        preamble_idx = np.argmax(correlation)
        max_corr = correlation[preamble_idx]
        
        print(f"Preamble correlation: {max_corr:.4f}")
        print(f"Preamble found at sample: {preamble_idx}")
        
        if max_corr < 0.1:
            print("WARNING: Weak preamble detection!")
            return None
        
        # Extract signal after preamble - exactly enough for 20 bits
        signal_start = preamble_idx + self.preamble_length
        samples_needed = self.n_bits * self.modulator.Ns  # Use actual samples per bit from modulator
        signal_end = signal_start + samples_needed
        
        if signal_end > len(signal):
            print(f"ERROR: Not enough samples! Need {signal_end}, have {len(signal)}")
            return None
        
        extracted = signal[signal_start:signal_end]
        print(f"Extracted {len(extracted)} samples ({len(extracted)/self.modulator.Ns:.0f} bits worth)")
        
        # Demodulate - should give exactly 20 bits
        bits = self.modulator.CSS_demodulate(extracted)
        
        print(f"Demodulated {len(bits)} bits")
        
        # Ensure exactly 20 bits
        if len(bits) != self.n_bits:
            print(f"ERROR: Got {len(bits)} bits, expected {self.n_bits}")
            return None
        
        return bits


def main():
    tester = SimpleAudioTest()
    
    print("\n" + "="*70)
    print("  SIMPLE 20-BIT TRANSMISSION TEST")
    print("="*70)
    print("\n[1] Send 20 bits")
    print("[2] Receive 20 bits")
    print("\n")
    
    choice = input("Select mode: ").strip()
    
    # Test pattern
    test_bits = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1])
    
    if choice == '1':
        tester.send(test_bits)
        print(f"Sent: {test_bits}\n")
    
    elif choice == '2':
        duration = float(input("Recording duration (default=10s): ").strip() or "10")
        received = tester.receive(duration)
        
        if received is not None:
            print(f"\nReceived: {received}")
            print(f"Expected: {test_bits}")
            
            if np.array_equal(received, test_bits):
                print("✓ SUCCESS! Perfect match!")
            else:
                errors = np.sum(received != test_bits)
                print(f"✗ {errors}/{len(test_bits)} bit errors")


if __name__ == "__main__":
    main()