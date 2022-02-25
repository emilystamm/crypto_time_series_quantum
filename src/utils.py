"""
===================================================
UTILS.PY
Helper functions.
===================================================
"""
# Imports
import pennylane as qml 

from model import num_qubits

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += (l - p) ** 2
    loss = loss / len(labels)
    return loss

def layer(weights):
    for wire in range(num_qubits - 1):
        qml.RX(weights[wire, 0], wires=[wire])
        qml.RY(weights[wire, 1], wires=[wire])
        qml.RZ(weights[wire, 2], wires=[wire])
        qml.CNOT(wires=[wire, wire + 1])

def full_layer(weights):
    for wire in range(num_qubits):
        qml.RX(weights[wire, 0], wires=[wire])
        qml.RY(weights[wire, 1], wires=[wire])
        qml.RZ(weights[wire, 2], wires=[wire])
    for wire in range(num_qubits-1):
        for wire2 in range(wire + 1, num_qubits):
            qml.CNOT(wires=[wire, wire2])

