"""
File: LempelZivCoder.py
Author: Hannes Stalder
Description: Implements Lempel-Ziv compression algorithm for source coding.
"""

from typing import List, Dict
import numpy as np

class LempelZivCoder:
    def __init__(self):
        self.index_bits = 20
        
    def encode(self, text: str) -> np.ndarray:
        if not text:
            raise ValueError("Input string cannot be empty")

        # Initialize dictionary with empty string and all unique characters from text
        dictionary = {"": 0}
        for char in set(text):
            if char not in dictionary:
                dictionary[char] = len(dictionary)
        
        encoded_indices: List[int] = []
        current_phrase = ""

        # Build dictionary dynamically and output indices
        for char in text:
            candidate_phrase = current_phrase + char
            # If the candidate phrase is already in the dictionary, keep building
            if candidate_phrase in dictionary:
                current_phrase = candidate_phrase
            # If not, output the index of current_phrase and add the new phrase to dictionary
            else:
                encoded_indices.append(dictionary[current_phrase])
                dictionary[candidate_phrase] = len(dictionary)
                current_phrase = char  # Start new phrase with current char

        # Handle any remaining phrase at the end
        if current_phrase:
            encoded_indices.append(dictionary[current_phrase])
        
        # Embed the initial dictionary into the binary message
        return self._encode_with_dictionary(dictionary, encoded_indices)
    
    def _encode_with_dictionary(self, full_dictionary: Dict[str, int], encoded_indices: List[int]) -> np.ndarray:
        bits = []
        
        # Get initial dictionary (only single characters, not the compound phrases)
        initial_chars = [char for char in full_dictionary.keys() if len(char) == 1]
        
        # Encode number of initial characters (8 bits = 0-255 unique chars)
        num_chars = len(initial_chars)
        bits.extend([int(b) for b in format(num_chars, '08b')])
        
        # Encode each character as 8-bit ASCII
        for char in initial_chars:
            bits.extend([int(b) for b in format(ord(char), '08b')])
        
        # Encode number of indices (32 bits, max 2^32 - 1 indices)
        num_indices = len(encoded_indices)
        bits.extend([int(b) for b in format(num_indices, '032b')])
        
        # Encode the indices
        for index in encoded_indices:
            bits.extend([int(b) for b in format(index, f'0{self.index_bits}b')])
        
        return np.array(bits, dtype=np.int8)
    
    def _decode_dictionary_and_indices(self, binary: np.ndarray) -> tuple[Dict[str, int], List[int]]:
        """Decode the initial dictionary and indices from binary format."""
        pos = 0
        
        # Read number of initial characters (first 8 bits)
        num_chars = int(''.join(str(int(b)) for b in binary[pos:pos+8]), 2)
        pos += 8
        
        # Read each character
        dictionary = {"": 0}  # Start with empty string at index 0
        for _ in range(num_chars):
            char_code = int(''.join(str(int(b)) for b in binary[pos:pos+8]), 2)
            char = chr(char_code)
            dictionary[char] = len(dictionary)
            pos += 8
        
        # Read number of indices (32 bits)
        num_indices = int(''.join(str(int(b)) for b in binary[pos:pos+32]), 2)
        pos += 32
        
        # Read exactly num_indices indices
        indices = []
        for _ in range(num_indices):
            index = int(''.join(str(int(b)) for b in binary[pos:pos+self.index_bits]), 2)
            indices.append(index)
            pos += self.index_bits
        
        return dictionary, indices
    
    def _indices_to_binary(self, indices: List[int]) -> np.ndarray:
        bits = [int(b) for index in indices 
                       for b in format(index, f'0{self.index_bits}b')]
        return np.array(bits, dtype=np.int8)

    def _binary_to_indices(self, binary: np.ndarray) -> List[int]:
        str_bits = binary.astype(str)
        return [int("".join(str_bits[i : i + self.index_bits]), 2)
                for i in range(0, len(binary), self.index_bits)]

    def decode(self, code: np.ndarray) -> str:
        # Decode the dictionary and indices from the binary message
        initial_dict, indices = self._decode_dictionary_and_indices(code)
        
        # Build the dictionary as a list for fast index lookup
        dictionary: List[str] = [""] * len(initial_dict)
        for phrase, idx in initial_dict.items():
            dictionary[idx] = phrase
        
        decoded_chunks: List[str] = []
        previous_phrase = ""

        for index in indices:
            if index < 0 or index >= len(dictionary):
                raise ValueError(f"Corrupt stream: bad index {index}")

            # Get the phrase for this index
            current_phrase = dictionary[index]
            decoded_chunks.append(current_phrase)
            
            # Add new dictionary entry: previous phrase + first char of current phrase
            if previous_phrase and current_phrase:
                new_entry = previous_phrase + current_phrase[0]
                dictionary.append(new_entry)
            
            previous_phrase = current_phrase

        return "".join(decoded_chunks)


# Example usage
if __name__ == "__main__":
    import os
    coder = LempelZivCoder()
    #text = "I want wanting to warrant waving a wand.
    file_path = os.path.join(os.path.dirname(__file__), "input_text_short.txt")
    file_handle = open(file_path, "r")
    text = file_handle.read()
    file_handle.close()
    
    code = coder.encode(text)
    decoded = coder.decode(code)

    print(f"Original: {text}")
    print(f"Encoded bits: {code}")
    print(f"Encoded length: {len(code)} bits")
    print(f"Decoded: {decoded}")
    print(f"Lossless: {decoded == text}")
