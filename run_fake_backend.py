from datetime import datetime
import json

# Use fake backend with a noise model emulating a real quantum computer (based on Qiskit's AerSimulator)

# https://qiskit.github.io/qiskit-aer/tutorials/3_building_noise_models.html

import numpy as np
import time
import socket
from qrisp.interface import QiskitBackend
from qiskit_aer import AerSimulator

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

# Example error probabilities
p_reset = 0.0001
p_meas = 0.0001
p_gate1 = 0.0001

# QuantumError objects
error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)

# Add errors to noise model
noise_bit_flip = NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

#print(noise_bit_flip)

def start_cpu_backend():
    ibm_simulator = AerSimulator(noise_model=noise_bit_flip)
    #ibm_simulator = AerSimulator()

    datetime_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    qpu_data = [
        {"id": "OneQubitGateErrorRate", "value": 1-error_gate1.probabilities[-1], "type": "float", "timestamp": datetime_string},
        {"id": "TwoQubitGateErrorRate", "value": 1-error_gate2.probabilities[-1], "type": "float", "timestamp": datetime_string},
        {"id": "SPAMErrorRate", "value": 1-error_meas.probabilities[-1], "type": "float", "timestamp": datetime_string},
        {"id": "ErrorCorrectionEnabled", "value": False, "type": "bool", "timestamp": datetime_string}
    ]

    # Writing data to a JSON file
    with open('qpu_data.json', 'w') as json_file:
        json.dump(qpu_data, json_file, indent=4)

    # Start Qrisp BackendServer
    ibm_backend = QiskitBackend(backend=ibm_simulator, port=42069)

    while True:
        time.sleep(1)

def port_already_in_use():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1",42069))
        except socket.error:
            return True
    return False

def main():
    if not port_already_in_use():
        start_cpu_backend()

if __name__ == "__main__":
    main()