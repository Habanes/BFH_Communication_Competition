"""
File: HuffmanCoder.py
Author: Hannes Stalder
Description: Implements Huffman coding for source compression
"""

import heapq
from collections import Counter
import numpy as np
import os

# In this module we use heqpq to efficielntly keep track of the nodes
# heapq lets us define a sorting function, then the heap automatically sorts itself accordingly
# this is very useful to keep track of the lowest probability nodes when builidng the tree

# the node class is used to buuild the tree
class Node:
    def __init__(self, freq, char=None, left=None, right=None):
        self.freq = freq
        self.char = char
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCoder:
    def __init__(self):
        self.tree = None
        self.codes = {}

    def build_encoding_map(self, text: str):
        if not text:
            self.tree = None
            self.codes = {}
            return
        
        # Counter returns a dict with each symbol and occurance count
        freq = Counter(text)
        
        # only one character
        if len(freq) == 1:
            char = list(freq.keys())[0] # simply get the first and only key
            self.tree = Node(freq[char], char=char) # we create a single node with the only char and freq we have
            self.codes = {char: '0'} # encoding is trivial
            return

        # Create all the nodes and sort them
        priority_queue = [Node(count, char) for char, count in freq.items()]
        heapq.heapify(priority_queue)
        
        while len(priority_queue) > 1:
            
            # we take the two lowest probability elements and form a new node
            left_node = heapq.heappop(priority_queue)
            right_node = heapq.heappop(priority_queue)
            
            # the new node has the combined occurence frequency
            merged_freq = left_node.freq + right_node.freq
            merged_node = Node(merged_freq, left=left_node, right=right_node)
            
            # We push the new node back to the heap. it is automatically sorted with to the __lt__ method.
            heapq.heappush(priority_queue, merged_node)

        self.tree = priority_queue[0]
        
        self.codes = {}
        self._generate_codes_recursive(self.tree, "", self.codes)

    def _generate_codes_recursive(self, node, current_code, codes_map):
        
        # if there is a character then we reached the end of a branch
        if node.char is not None:
            codes_map[node.char] = current_code
            return
        
        # if not then we are at a juncion node
        if node.left:
            self._generate_codes_recursive(node.left, current_code + "0", codes_map)
        
        if node.right:
            self._generate_codes_recursive(node.right, current_code + "1", codes_map)

    def encode(self, text: str) -> np.ndarray:
        # Build encoding map from the text
        self.build_encoding_map(text)
        
        if not self.codes:
            raise ValueError("Encoding map not built.")
        
        # Build the bit string for the message
        bit_string = "".join(self.codes[char] for char in text)
        
        # Embed the codes dictionary in the message
        return self._encode_with_codebook(self.codes, bit_string)
    
    def _encode_with_codebook(self, codes: dict, bit_string: str) -> np.ndarray:
        """Encode the Huffman codebook and message into binary format.
        
        Format:
        - First 8 bits: number of unique characters
        - For each character:
          - 8 bits: ASCII character code
          - 8 bits: length of Huffman code (max 255)
          - N bits: the Huffman code itself
        - 32 bits: length of the encoded message
        - Remaining bits: the encoded message
        """
        bits = []
        
        # Encode number of characters (8 bits)
        num_chars = len(codes)
        bits.extend([int(b) for b in format(num_chars, '08b')])
        
        # Encode each character and its Huffman code
        for char, code in codes.items():
            # Character as 8-bit ASCII
            bits.extend([int(b) for b in format(ord(char), '08b')])
            
            # Length of Huffman code (8 bits, max 255)
            code_length = len(code)
            bits.extend([int(b) for b in format(code_length, '08b')])
            
            # The Huffman code itself
            bits.extend([int(b) for b in code])
        
        # Encode the length of the message (32 bits, max 2^32 - 1 bits)
        message_length = len(bit_string)
        bits.extend([int(b) for b in format(message_length, '032b')])
        
        # Encode the actual message
        bits.extend([int(b) for b in bit_string])
        
        return np.array(bits, dtype=np.int8)
    
    def _decode_codebook_and_message(self, binary: np.ndarray) -> tuple[dict, str]:
        """Decode the Huffman codebook and message from binary format.
        
        Returns:
            tuple: (codes_dict, encoded_message_bits)
        """
        pos = 0
        
        # Read number of characters (first 8 bits)
        num_chars = int(''.join(str(int(b)) for b in binary[pos:pos+8]), 2)
        pos += 8
        
        # Read each character and its code
        codes = {}
        for _ in range(num_chars):
            # Read character (8 bits)
            char_code = int(''.join(str(int(b)) for b in binary[pos:pos+8]), 2)
            char = chr(char_code)
            pos += 8
            
            # Read code length (8 bits)
            code_length = int(''.join(str(int(b)) for b in binary[pos:pos+8]), 2)
            pos += 8
            
            # Read the Huffman code
            code = ''.join(str(int(b)) for b in binary[pos:pos+code_length])
            codes[char] = code
            pos += code_length
        
        # Read message length (32 bits)
        message_length = int(''.join(str(int(b)) for b in binary[pos:pos+32]), 2)
        pos += 32
        
        # Read exactly message_length bits for the message
        message_bits = ''.join(str(int(b)) for b in binary[pos:pos+message_length])
        
        return codes, message_bits

    def decode(self, encoded: np.ndarray) -> str:
        """Decode a NumPy array of bits back to text."""
        # Decode the codebook and message from the binary data
        codes, message_bits = self._decode_codebook_and_message(encoded)
        
        # Build reverse mapping (code -> char)
        reverse_codes = {code: char for char, code in codes.items()}
        
        # Handle single character case
        if len(codes) == 1:
            char = list(codes.keys())[0]
            return char * len(message_bits)
        
        # Decode the message
        decoded = []
        current_code = ""
        
        for bit in message_bits:
            current_code += bit
            
            if current_code in reverse_codes:
                decoded.append(reverse_codes[current_code])
                current_code = ""
        
        return "".join(decoded)


# Example usage
if __name__ == "__main__":
    
    coder = HuffmanCoder()
    # text = "hallihallo"
    file_path = os.path.join(os.path.dirname(__file__), "input_text_short.txt")
    file_handle = open(file_path, "r")
    text = file_handle.read()
    file_handle.close()
    
    # Encode (no need to call build_encoding_map separately)
    encoded = coder.encode(text)
    
    # Decode (decoder doesn't need the original text!)
    decoder = HuffmanCoder()  # Fresh decoder
    decoded = decoder.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded bits: {encoded}")
    print(f"Encoded length: {len(encoded)} bits")
    print(f"Decoded: {decoded}")
    print(f"Lossless: {decoded == text}")
