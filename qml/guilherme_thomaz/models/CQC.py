import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.app import ArrayRecord

def create_cqc_quantum_circuit(n_qubits: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def quantum_circuit(inputs, weights):
        """Quantum circuit for the QNN layer."""
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return quantum_circuit

class CQC(nn.Module):

    def __init__(self, num_classes: int = 10, n_qubits: int = 4, n_layers: int = 3):
        super(CQC, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # CNN feature extraction layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Classical dense layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_qubits)

        # Create quantum circuit and layer
        quantum_circuit = create_cqc_quantum_circuit(n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnn = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # Classical post-processing
        self.fc_out = nn.Linear(n_qubits, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten for dense layers
        x = x.view(-1, 16 * 5 * 5)

        # Classical dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        x = torch.relu(x)

        # Quantum layer
        x = self.qnn(x)

        # Output layer
        x = self.fc_out(x)
        return x

def get_initial_cqc_array(context):
    n_qubits: int = context.run_config.get("n-qubits", 4)
    n_layers: int = context.run_config.get("n-layers", 3)
    print(f"  - Number of qubits: {n_qubits}", flush=True)
    print(f"  - Number of layers: {n_layers}", flush=True)
    global_model = CQC(num_classes=10, n_qubits=n_qubits, n_layers=n_layers)
    print(f"Initialized quantum neural network with {sum(p.numel() for p in global_model.parameters())} parameters", flush=True)
    arrays = ArrayRecord(global_model.state_dict())
    return arrays