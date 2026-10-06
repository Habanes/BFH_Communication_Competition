"""
File: TransmissionModule_CSS.py
Author: Modified by AI Assistant
Description: Transmission module using CSS Modulation instead of QPSK with hardware interface support
"""

from typing import Optional, Literal
import numpy as np
import os
from transmission_competition.HuffmanCoder import HuffmanCoder
from transmission_competition.LempelZivCoder import LempelZivCoder
from transmission_competition.HammingCoder74 import HammingCoder74
from transmission_competition.EntropyCalculator import EntropyCalculator
from transmission_competition.CSSModulator import CSSModulator  # Changed from Modulation
from transmission_competition.HardwareInterface import HardwareInterface


class TransmissionModule_CSS:
    def __init__(self, input_string: str = "", use_lempel_ziv: bool = False, per_bit_error_rate: float = 0.00, 
                 snr_db: Optional[float] = None, mode: Literal["sender", "receiver", "simulation"] = "simulation",
                 use_hardware: bool = False):
        """
        Initialize TransmissionModule with CSS modulation.
        
        :param input_string: Text to transmit (required for sender/simulation modes)
        :param use_lempel_ziv: Use Lempel-Ziv compression instead of Huffman
        :param per_bit_error_rate: Bit error rate for channel simulation
        :param snr_db: Signal-to-noise ratio in dB (None for no noise)
        :param mode: Operation mode - "sender", "receiver", or "simulation"
        :param use_hardware: Enable hardware interface for real audio transmission
        """
        
        self.mode = mode
        self.use_hardware = use_hardware
        self.input_string: str = input_string
        
        self.per_bit_error_rate = per_bit_error_rate
        self.snr_db = snr_db
        
        # Initialize hardware interface if needed
        if self.use_hardware:
            self.hardware = HardwareInterface()
        else:
            self.hardware = None
        
        # Initialise source coder class
        if use_lempel_ziv:
            self.sourceCoder = LempelZivCoder()
        else:
            self.sourceCoder = HuffmanCoder()
            if input_string:  # Only build encoding map if we have input
                self.sourceCoder.build_encoding_map(input_string)
        
        # Initialise channel coder class
        self.channelCoder = HammingCoder74()
        
        # Initialise CSS modulation class
        # Use slower symbol rate for hardware transmission to improve reliability
        if self.use_hardware:
            # Slower: 50ms per symbol for more robust audio transmission
            self.modulator = CSSModulator(fs=48000.0, T_symbol=0.05, f_start=1000.0, bandwidth=3000.0)
        else:
            # Faster for simulation: 10ms per symbol (default)
            self.modulator = CSSModulator()  # Default: T_symbol=0.01
    
        # Initialize variables that will be set during processing
        self.source_coded = None
        self.channel_coded = None
        self.modulated_signal = None
        self.transmitted_signal = None
        self.demodulated_bits = None
        self.transmitted = None
        self.error_description = None
        self.channel_decoded = None
        self.output_string = None
        self.lossless = None
        self.entropyCalculator = None
        
        # If in simulation mode, run full pipeline immediately
        if self.mode == "simulation" and input_string:
            self._run_simulation()
    
    def _run_simulation(self):
        """Run complete simulation pipeline (encode -> modulate -> demodulate -> decode)"""
        self.entropyCalculator = EntropyCalculator(self.input_string)
        
        # Source encode the string
        self.source_coded = self.sourceCoder.encode(self.input_string)

        # Channel encode the source_coded string
        self.channel_coded = self.channelCoder.encode(self.source_coded)

        # NOTE: CSS modulation handles 1 bit per symbol, so no padding needed like QPSK
        # Modulate the channel_coded bits using CSS
        self.modulated_signal = self.modulator.CSS_modulate(self.channel_coded)
        
        # Add noise if SNR is specified
        if self.snr_db is not None:
            self.transmitted_signal = self.modulator.add_awgn_noise(self.modulated_signal, self.snr_db)
        else:
            self.transmitted_signal = self.modulated_signal
        
        # Demodulate the transmitted signal
        self.demodulated_bits = self.modulator.CSS_demodulate(self.transmitted_signal)
        
        # Simulate transmission through a noisy channel (bit-level errors)
        self.transmitted, self.error_description = self.channelCoder.channel_simulator(self.demodulated_bits, self.per_bit_error_rate)

        # Channel decode the transmitted string
        self.channel_decoded = self.channelCoder.decode(self.transmitted)

        # Decode the source encoding
        try:
            self.output_string = self.sourceCoder.decode(self.channel_decoded)
        except ValueError as e:
            self.output_string = f"Decoding failed: {str(e)}"
            
        self.lossless = self.input_string == self.output_string
    
    def start_transmission(self):
        """
        Start transmission (sender mode).
        Encodes input string and transmits via hardware or returns modulated signal.
        """
        if not self.input_string:
            raise ValueError("No input string provided for transmission")
        
        if self.mode != "sender" and self.mode != "simulation":
            print(f"Warning: Module is in '{self.mode}' mode, switching to 'sender' mode")
            self.mode = "sender"
        
        # Initialize entropy calculator
        self.entropyCalculator = EntropyCalculator(self.input_string)
        
        # Source encode
        self.source_coded = self.sourceCoder.encode(self.input_string)
        
        # Channel encode
        self.channel_coded = self.channelCoder.encode(self.source_coded)
        
        # Modulate
        self.modulated_signal = self.modulator.CSS_modulate(self.channel_coded)
        
        print(f"Transmission prepared: {len(self.channel_coded)} bits -> {len(self.modulated_signal)} samples")
        
        # Transmit via hardware if enabled
        if self.use_hardware and self.hardware is not None:
            print("Starting hardware transmission...")
            self.hardware.start_transmission(self.modulated_signal)
            print("Transmission complete!")
        else:
            print("Simulation mode: modulated signal ready (use .modulated_signal)")
        
        return self.modulated_signal
    
    def start_reception(self, received_signal: Optional[np.ndarray] = None, 
                       reception_duration: Optional[float] = None) -> str:
        """
        Start reception (receiver mode).
        Receives signal from hardware or uses provided signal, then demodulates and decodes.
        
        :param received_signal: Pre-recorded signal (if not using hardware)
        :param reception_duration: Duration to record if using hardware [s]
        :return: Decoded output string
        """
        if self.mode != "receiver" and self.mode != "simulation":
            print(f"Warning: Module is in '{self.mode}' mode, switching to 'receiver' mode")
            self.mode = "receiver"
        
        # Get received signal
        if self.use_hardware and self.hardware is not None:
            if reception_duration is None:
                raise ValueError("reception_duration must be specified when using hardware")
            
            print("Starting hardware reception...")
            self.transmitted_signal = self.hardware.start_reception(reception_duration)
            
            if self.transmitted_signal is None:
                raise RuntimeError("Reception failed - preamble not detected")
            
            print(f"Received {len(self.transmitted_signal)} samples")
        else:
            if received_signal is None:
                raise ValueError("received_signal must be provided when not using hardware")
            self.transmitted_signal = received_signal
        
        # Demodulate
        self.demodulated_bits = self.modulator.CSS_demodulate(self.transmitted_signal)
        
        # Channel decode
        self.channel_decoded = self.channelCoder.decode(self.demodulated_bits)
        
        # Source decode
        try:
            self.output_string = self.sourceCoder.decode(self.channel_decoded)
        except ValueError as e:
            self.output_string = f"Decoding failed: {str(e)}"
        
        print(f"Reception complete: decoded '{self.output_string}'")
        return self.output_string

    def __repr__(self):
        # Helper function to convert numpy array to bitstring for display
        def array_to_str(arr):
            if isinstance(arr, np.ndarray):
                return ''.join(map(str, arr.astype(int)))
            return str(arr)
        
        mode_str = f"Mode: {self.mode}" + (" (with hardware)" if self.use_hardware else " (simulation)")
        
        return (f"TransmissionModule_CSS:\n"
                f"*****SUMMARY*******************************************************************\n\n"
                f"**  {mode_str}\n\n"
                f"**  Input String: '{self.input_string if self.input_string else 'N/A'}'\n\n"
                f"**  Entropy Calculations: {self.entropyCalculator if self.entropyCalculator else 'N/A'}\n\n"                
                f"**  Source Coded: \n'{array_to_str(self.source_coded) if self.source_coded is not None else 'N/A'}'\n\n"
                f"**  Channel Coded: \n'{array_to_str(self.channel_coded) if self.channel_coded is not None else 'N/A'}'\n\n"
                f"**  CSS Modulated Signal: {len(self.modulated_signal) if self.modulated_signal is not None else 'N/A'} samples\n\n"
                f"**  SNR (dB): {self.snr_db if self.snr_db is not None else 'N/A (no noise)'}\n\n"
                f"**  Demodulated Bits: \n'{array_to_str(self.demodulated_bits) if self.demodulated_bits is not None else 'N/A'}'\n\n"
                f"**  Transmitted: \n'{array_to_str(self.transmitted) if self.transmitted is not None else 'N/A'}'\n\n"
                f"**  Error Description: '{self.error_description if self.error_description else 'N/A'}'\n\n"
                f"**  Channel Decoded: \n'{array_to_str(self.channel_decoded) if self.channel_decoded is not None else 'N/A'}'\n\n"
                f"**  Output String: '{self.output_string if self.output_string else 'N/A'}'\n\n"
                f"**  Lossless: {self.lossless if self.lossless is not None else 'N/A'}\n\n"
                f"******************************************************************************\n")
        
        
if __name__ == "__main__":
    print("="*80)
    print("Testing TransmissionModule_CSS with Hardware Interface")
    print("="*80)
    
    # Test 1: Simulation mode (original behavior)
    print("\n--- Test 1: Simulation mode (no hardware) ---")
    text1 = "hello,world"
    tm1 = TransmissionModule_CSS(input_string=text1, use_lempel_ziv=False, snr_db=None, mode="simulation")
    print(f"Input:  '{tm1.input_string}'")
    print(f"Output: '{tm1.output_string}'")
    print(f"Lossless: {tm1.lossless}")
    
    # Test 2: Sender mode
    print("\n--- Test 2: Sender mode (prepare transmission) ---")
    tm_sender = TransmissionModule_CSS(input_string=text1, use_lempel_ziv=False, mode="sender")
    modulated = tm_sender.start_transmission()
    print(f"Modulated signal length: {len(modulated)} samples ({len(modulated)/tm_sender.modulator.fs:.3f} seconds)")
    
    # Test 3: Receiver mode (simulate reception)
    print("\n--- Test 3: Receiver mode (receive and decode) ---")
    tm_receiver = TransmissionModule_CSS(use_lempel_ziv=False, mode="receiver")
    # Simulate receiving the same signal (in practice this would come from hardware)
    received_text = tm_receiver.start_reception(received_signal=modulated)
    print(f"Received text: '{received_text}'")
    print(f"Match: {received_text == text1}")
    
    # Test 4: With noise in simulation
    print("\n--- Test 4: Simulation with noise (SNR=15dB) ---")
    tm4 = TransmissionModule_CSS(input_string=text1, use_lempel_ziv=False, snr_db=15.0, mode="simulation")
    print(f"Input:  '{tm4.input_string}'")
    print(f"Output: '{tm4.output_string}'")
    print(f"Lossless: {tm4.lossless}")
    
    # Test 5: Hardware mode (requires user interaction)
    print("\n--- Test 5: Hardware transmission test (optional) ---")
    print("This test uses actual audio hardware (speaker/microphone)")
    
    test_hardware = input("Do you want to test hardware transmission? (y/n): ").strip().lower() == 'y'
    
    if test_hardware:
        print("\n=== HARDWARE TRANSMISSION TEST ===")
        test_text = "test"
        
        # Create sender with hardware
        print("\n1. Creating sender...")
        hw_sender = TransmissionModule_CSS(input_string=test_text, mode="sender", use_hardware=True)
        
        input("\nPress Enter to start transmission...")
        hw_sender.start_transmission()
        
        # Calculate expected duration after transmission is prepared
        signal_duration = len(hw_sender.channel_coded) * hw_sender.modulator.Ts
        total_duration = signal_duration + hw_sender.hardware.preamble_duration + hw_sender.hardware.silence_duration
        reception_duration = total_duration + 0.5  # Add buffer
        
        print(f"\nReceiver should record for ~{reception_duration:.2f} seconds")
        
        print("\n2. Creating receiver...")
        hw_receiver = TransmissionModule_CSS(mode="receiver", use_hardware=True)
        
        input("\nPress Enter to start reception...")
        received_hw = hw_receiver.start_reception(reception_duration=reception_duration)
        
        print(f"\nOriginal text: '{test_text}'")
        print(f"Received text: '{received_hw}'")
        print(f"Success: {received_hw == test_text}")
    else:
        print("Hardware test skipped.")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
