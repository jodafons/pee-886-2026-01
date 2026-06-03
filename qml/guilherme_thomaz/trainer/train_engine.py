from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.BinaryQCNN import BinaryQCNN
from models.CQC import CQC
from evaluation.eval_engine import test_cqc, test_binqcnn
import pennylane as qml

def handle_cqc_train_call(msg, context, trainloader, valloader):

    n_qubits = context.run_config.get("n-qubits", 4)
    n_layers = context.run_config.get("n-layers", 3)

    # Load the model and initialize it with the received weights
    model = CQC(num_classes=10, n_qubits=n_qubits, n_layers=n_layers)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Training data size: {len(trainloader.dataset)}")
    print(f"Validation data size: {len(valloader.dataset)}")

    # Call the training function
    results = train_cqc(
        model,
        trainloader,
        valloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )
    return model, results

def handle_binqcnn_train_call(msg, context, trainloader, valloader):

    # Load the model and initialize it with the received weights
    model = BinaryQCNN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Training data size: {len(trainloader.dataset)}")
    print(f"Validation data size: {len(valloader.dataset)}")

    # Call the training function
    results = train_binqcnn(
        model,
        trainloader,
        valloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )
    return model, results

def train_cqc(
    net: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, float]:
    """Train the quantum neural network."""
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    running_loss = 0.0
    for _ in range(epochs):
        for batch_idx, batch in enumerate(trainloader):
            data = batch["img"].to(device)
            target = torch.as_tensor(batch["label"], dtype=torch.long, device=device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # Evaluate on validation set
    val_loss, val_accuracy = test_cqc(net, valloader, device)

    avg_train_loss = running_loss / (epochs * len(trainloader))

    return {
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "num_examples": len(trainloader.dataset),
    }

def train_binqcnn(
    net: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, float]:
    
    net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    
    running_loss = 0
    for epoch in range(epochs): 
        for batch_idx, batch in enumerate(trainloader):
            data = batch["img"].to(device)
            target = torch.as_tensor(batch["label"], dtype=torch.double, device=device)
            
            optimizer.zero_grad()
            out = net(data)
            
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            #preds = torch.round(torch.sigmoid(out))
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(trainloader)}, Loss: {loss.item():.4f}", flush=True)

    # Evaluate on validation set
    val_loss, val_accuracy = test_binqcnn(net, valloader, device)

    avg_train_loss = running_loss / (epochs * len(trainloader))

    return {
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "num_examples": len(trainloader.dataset),
    }