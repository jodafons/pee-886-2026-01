import pennylane as qml
import torch
import torch.nn as nn
from time import time
from ansatz import pool_ansatz, conv_neighbor
from autoencoders import AE, load_mnist

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def append_log(log_file, log_entry):
    print(log_entry)
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")


n_qubits = 8
device_name = "lightning.gpu" if torch.cuda.is_available() else "default.qubit"
dev = qml.device(device_name, wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def hierarchical_qcnn_neighbor(inputs, conv_weights, pool_weights, conv_layer=conv_neighbor, pool_layer=pool_ansatz):
    qml.AngleEmbedding(features=inputs, wires=range(n_qubits), rotation='X')

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


def get_hierarchical_circuit(conv_layer, pool_layer):
    return lambda x, y ,z : hierarchical_qcnn_neighbor(x,
                                                       y,
                                                       z,
                                                       conv_layer=conv_layer,
                                                       pool_layer=pool_layer)


class HierarchicalQCNN(nn.Module):
    def __init__(self, circuit):
        super().__init__()
        self.conv_weights = nn.Parameter(0.1 * torch.randn(7, 2))
        self.pool_weights = nn.Parameter(0.1 * torch.randn(7, 2))
        self.circuit = circuit

    def forward(self, x):
        q_out = self.circuit(x, self.conv_weights, self.pool_weights)
        return 5 * q_out.view(-1, 1).float()


class MNISTEnsemble(nn.Module):
    def __init__(self, circuit, encoder):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.models = nn.ModuleList([HierarchicalQCNN(circuit) for _ in range(10)])

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        features = torch.sigmoid(self.encoder(x))
        outputs = [model(features) for model in self.models]
        return 5 * torch.cat(outputs, dim=1)

    def test(self, data_loader, device='cpu'):
        self.eval()
        self.to(device)
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                logits = self(batch_x)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_y.size(0)
                
        accuracy = total_correct / total_samples
        return accuracy

    def save_confusion_matrix(self, data_loader, device='cpu', filename='confusion_matrix.png'):
        self.eval()
        self.to(device)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                logits = self(batch_x)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(filename)
        print(f"Confusion matrix saved to {filename}")

    def save_weights(self, filename='ensemble_weights.pth'):
        torch.save(self.state_dict(), filename)
        print(f"Weights saved to {filename}")

    def fit(self, train_loader, test_loader, epochs=10, lr=0.01, device='cpu', verbose=True):
        self.to(device)
        optimizers = [torch.optim.Adam(model.parameters(), lr=lr) for model in self.models]
        loss_fn = nn.BCEWithLogitsLoss()
        
        print(f"Starting ensemble training on {device}...")
        for epoch in range(epochs):
            init_time = time()
            self.train()
            total_loss = 0
            
            ii = 0
            for batch_x, batch_y in train_loader:
                ii += 1
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                with torch.no_grad():
                    batch_x = batch_x.view(-1, 28 * 28)
                    features = torch.sigmoid(self.encoder(batch_x))
                
                batch_loss = 0
                for i in range(10):
                    optimizers[i].zero_grad()
                    target = (batch_y == i).float().view(-1, 1)
                    out = self.models[i](features)
                    loss = loss_fn(out, target)
                    loss.backward()
                    optimizers[i].step()
                    batch_loss += loss.item()
                
                total_loss += batch_loss
            
            avg_loss = total_loss / (len(train_loader) * 10)
            test_acc = self.test(test_loader, device=device)
            
            if verbose:
                log_msg = f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.4f} | Time: {time() - init_time:.2f}s"
                append_log('./hierarchical_ae_results.txt', log_msg)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, test_loader = load_mnist(batch_size=32, n_train=5120*2)

    ae = AE()
    ae.load_state_dict(torch.load('artifacts/ae_best.pth', map_location=device, weights_only=True))
    encoder = ae.encoder

    circuit = get_hierarchical_circuit(conv_layer=conv_neighbor, pool_layer=pool_ansatz)

    model = MNISTEnsemble(circuit=circuit, encoder=encoder).to(device)
    model.fit(
        train_loader=train_loader, 
        test_loader=test_loader, 
        epochs=20, 
        lr=0.01,
        device=device
    )

    model.save_weights('artifacts/hierarchical_ae_last.pth')


if __name__ == '__main__':
    main()
