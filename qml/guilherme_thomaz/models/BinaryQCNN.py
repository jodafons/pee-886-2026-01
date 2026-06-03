import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from flwr.app import ArrayRecord
from matplotlib import pyplot as plt

def print_circuit(circuit, name):
    inputs = torch.randn(256)
    weights = nn.Parameter(0.1 * torch.randn(2, 7, 2))
    #print(qml.draw(circuit)(inputs, conv_weights, pool_weights))
    fig, ax = qml.draw_mpl(circuit)(inputs, weights)
    plt.savefig(f"{name}.png")
    print(f"Circuit diagram saved to {name}.png")

def create_binqcnn_quantum_circuit():
    n_qubits = 8
    device_name = "lightning.gpu" if torch.cuda.is_available() else "default.qubit"
    dev = qml.device(device_name, wires=n_qubits)
        
    def pool_layer(weights, wires):
        qml.CRZ(weights[0], wires=[wires[1], wires[0]])
        qml.PauliX(wires=wires[1])
        qml.CRX(weights[1], wires=[wires[1], wires[0]])

    def conv_layer(weights, wires):
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def quantum_circuit(inputs, weights):
        qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)

        conv_weights = weights[0]
        pool_weights = weights[1]

        conv_layer(conv_weights[0], wires=[0, 1])
        pool_layer(pool_weights[0], wires=[0, 1])
        conv_layer(conv_weights[1], wires=[2, 3])
        pool_layer(pool_weights[1], wires=[2, 3])
        conv_layer(conv_weights[2], wires=[4, 5])
        pool_layer(pool_weights[2], wires=[4, 5])
        conv_layer(conv_weights[3], wires=[6, 7])
        pool_layer(pool_weights[3], wires=[6, 7])

        conv_layer(conv_weights[4], wires=[0, 2])
        pool_layer(pool_weights[4], wires=[0, 2])
        conv_layer(conv_weights[5], wires=[4, 6])
        pool_layer(pool_weights[5], wires=[4, 6])

        conv_layer(conv_weights[6], wires=[0, 4])
        pool_layer(pool_weights[6], wires=[0, 4])

        return qml.expval(qml.PauliZ(0))

    return quantum_circuit

class BinaryQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Registra um único parâmetro com formato [2, 7, 2]
        # Índice 0 = conv_weights | Índice 1 = pool_weights
        self.weights = nn.Parameter(0.1 * torch.randn(2, 7, 2))
        self.circuit = create_binqcnn_quantum_circuit()
        print_circuit(self.circuit, "binqcnn_circuit")

    def forward(self, x):
        q_out = self.circuit(x, self.weights)
        return q_out.double()
    
def get_initial_binqcnn_array(context):
    global_model = BinaryQCNN()
    print(f"Initialized quantum neural network with {sum(p.numel() for p in global_model.parameters())} parameters", flush=True)
    arrays = ArrayRecord(global_model.state_dict())
    return arrays