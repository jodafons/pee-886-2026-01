import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from time import time
from ansatz import conv_neighbor
from autoencoders import AE, load_mnist

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


def append_log(log_file, log_entry):
    print(log_entry)
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")


n_qubits = 8
device_name = "lightning.gpu" if torch.cuda.is_available() else "default.qubit"
dev = qml.device(device_name, wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def qcnn(inputs,
         conv_weights,
         conv_layer=conv_neighbor):
    qml.AngleEmbedding(features=inputs, wires=range(n_qubits), rotation='X')

    conv_layer(conv_weights[0], wires=[0, 1])
    conv_layer(conv_weights[1], wires=[2, 3])
    conv_layer(conv_weights[2], wires=[4, 5])
    conv_layer(conv_weights[3], wires=[6, 7])

    conv_layer(conv_weights[4], wires=[1, 2])
    conv_layer(conv_weights[5], wires=[3, 4])
    conv_layer(conv_weights[6], wires=[5, 6])
    conv_layer(conv_weights[7], wires=[7, 0])

    return (qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(2)),
            qml.expval(qml.PauliZ(3)),
            qml.expval(qml.PauliZ(4)),
            qml.expval(qml.PauliZ(5)),
            qml.expval(qml.PauliZ(6)),
            qml.expval(qml.PauliZ(7)))


def get_no_pool_circuit(conv_layer):
    return lambda x, y: qcnn(x,
                             y,
                             conv_layer=conv_layer)


class QCNN(nn.Module):
    def __init__(self, circuit, encoder):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.conv_weights = nn.Parameter(0.1 * torch.randn(8, 2))

        self.fc = nn.Linear(8, 10)
        self.circuit = circuit


    def forward(self, x):
        x = x.view(-1, 28 * 28)
        features = self.encoder(x)
        features = F.sigmoid(features)
        q_out = self.circuit(features, self.conv_weights)
        if isinstance(q_out, (list, tuple)):
            q_out = torch.stack(q_out, dim=1)
        return self.fc(q_out.float())

    def test(self, data_loader, device='cpu'):
        self.eval()
        self.to(device)
        loss_fn = nn.CrossEntropyLoss()
        
        total_loss, total_correct = 0, 0
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                out = self(batch_x)
                loss = loss_fn(out, batch_y)
                
                total_loss += loss.item()
                preds = out.argmax(dim=1)
                total_correct += (preds == batch_y).sum().item()
                
        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / len(data_loader.dataset)
        return avg_loss, accuracy

    def save_confusion_matrix(self, data_loader, device='cpu', filename='QCNN_confusion_matrix.png'):
        self.eval()
        self.to(device)
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                out = self(batch_x)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(filename)
        plt.close()
        print(f"Confusion matrix saved to {filename}")

    def fit(self, train_loader, test_loader, epochs=50, lr=0.01, device='cpu', verbose=True):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f"Starting training on {device}...")
        for epoch in range(epochs):
            init_time = time()
            
            self.train()
            train_loss, train_correct = 0, 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                out = self(batch_x)
                
                loss = loss_fn(out, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = out.argmax(dim=1)
                train_correct += (preds == batch_y).sum().item()

            scheduler.step()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / len(train_loader.dataset)
            
            test_loss, test_acc = self.test(test_loader, device=device)
            
            if verbose:
                append_log('./qcnn.txt', f"""Epoch {epoch+1}
                                          Time: {time() - init_time:.2f}s
                                          Train: loss: {avg_train_loss:.4f}, acc: {train_acc:.4f}
                                          Test: loss: {test_loss:.4f}, acc: {test_acc:.4f}\n""")

    def save_weights(self, filename='qcnn_weights.pth'):
        torch.save(self.state_dict(), filename)
        print(f"Weights saved to {filename}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, test_loader = load_mnist(batch_size=32, n_train=5120*2)

    ae = AE()
    ae.load_state_dict(torch.load('artifacts/ae_best.pth', map_location=device, weights_only=True))
    encoder = ae.encoder

    circuit = get_no_pool_circuit(conv_layer=conv_neighbor)

    model = QCNN(circuit=circuit, encoder=encoder).to(device)
    model.fit(
        train_loader=train_loader, 
        test_loader=test_loader, 
        epochs=20, 
        lr=0.1, 
        device=device
    )
    model.save_weights('artifacts/qcnn_last.pth')


if __name__ == '__main__':
    main()
