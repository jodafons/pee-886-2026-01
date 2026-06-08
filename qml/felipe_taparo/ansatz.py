import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


def append_log(log_file, log_entry):
    print(log_entry)
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")


n_qubits = 8
device_name = "lightning.gpu" if torch.cuda.is_available() else "default.qubit"
dev = qml.device(device_name, wires=n_qubits)


def pool_ansatz(weights, wires):
    qml.CRZ(weights[0], wires=[wires[1], wires[0]])
    qml.PauliX(wires=wires[1])
    qml.CRX(weights[1], wires=[wires[1], wires[0]])


def conv_neighbor(weights, wires):
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def conv_neighbor_U3(weights, wires):
    qml.U3(weights[0], weights[1], weights[2], wires=wires[0])
    qml.U3(weights[3], weights[4], weights[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(weights[6], wires=wires[0])
    qml.RZ(weights[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(weights[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(weights[9], weights[10], weights[11], wires=wires[0])
    qml.U3(weights[12], weights[13], weights[14], wires=wires[1])


def print_circuit(circuit, name, conv_weights_shape, pool_weights_shape):
    inputs = torch.randn(256)
    conv_weights = torch.randn(*conv_weights_shape)
    pool_weights = torch.randn(*pool_weights_shape)
    print(qml.draw(circuit)(inputs, conv_weights, pool_weights))
    fig, ax = qml.draw_mpl(circuit)(inputs, conv_weights, pool_weights)
    fig.savefig(f"{name}.png")
    print(f"Circuit diagram saved to {name}.png")
