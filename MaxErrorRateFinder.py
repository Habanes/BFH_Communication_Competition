"""
File: MaxErrorRateFinder.py
Author: Hannes Stalder
Description: Finds the maximum bit error rate at which the transmission pipeline
             can still successfully decode messages with a specified confidence level.
"""

from TransmissionModule import TransmissionModule

def test_pipeline_for_error_rate(input_string, error_rate, confidence):
    
    NUMBER_OF_TESTS = 100
    tests_succeeded = 0
    tests_failed = 0
    
    for i in range(NUMBER_OF_TESTS):
        TestTransmissionModule = TransmissionModule(input_string=input_string, per_bit_error_rate=error_rate, use_lempel_ziv=False)
        if TestTransmissionModule.lossless:
            tests_succeeded += 1
        else:
            tests_failed += 1
            
    success_ratio = tests_succeeded/NUMBER_OF_TESTS
    print(f"Success rate: {success_ratio}")
    
    if tests_failed == 0 or success_ratio >= confidence:
        return True
    
    return False

if __name__ == "__main__":
    
    import os

    file_path = os.path.join(os.path.dirname(__file__), "transmission_competition/input_text_short.txt")
    file_handle = open(file_path, "r")
    input_string = file_handle.read()
    file_handle.close()
    
    # Start from 0 and work upwards to find the maximum error rate
    current_error_rate = 0.0007
    confidence = 0.95
    
    increase_per_step = 0.00001  # Start with small steps
    max_error_rate = 0.0  # Track the maximum successful error rate
    
    while current_error_rate <= 1.0:
        print(f"Testing error rate: {current_error_rate:.6f}")
        current_test_result = test_pipeline_for_error_rate(input_string, current_error_rate, confidence)
        
        if current_test_result == True:
            max_error_rate = current_error_rate
            print(f"Success at error rate: {current_error_rate:.6f}")
            current_error_rate += increase_per_step
        else:
            print(f"Failed at error rate: {current_error_rate:.6f}")
            break
    
    print(f"\nTEST RESULT: MAX_ERROR_RATE = {max_error_rate:.6f}")