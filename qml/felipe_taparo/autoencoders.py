import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader


def append_log(log_file, log_entry):
    print(log_entry)
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def test(self, test_loader, device='cuda'):
        self.eval()
        self.to(device)
        loss_function = nn.BCELoss()
        total_loss = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.view(-1, 28 * 28).to(device)
                reconstructed = self.forward(images)
                loss = loss_function(reconstructed, images)
                total_loss += loss.item()
        return total_loss / len(test_loader), reconstructed

    def fit(self, train_loader, test_loader, epochs=50, lr=1e-3, device='cuda', verbose=True):
        self.to(device)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-8)
        loss_function = nn.BCELoss()
        last_loss = 1000
        for epoch in range(epochs):
            for images, _ in train_loader:
                images = images.view(-1, 28 * 28).to(device)
                
                reconstructed = self.forward(images)
                loss = loss_function(reconstructed, images)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if loss.item() < last_loss:
                torch.save(self.state_dict(), './artifacts/ae_best.pth')
                last_loss = loss.item()

            append_log('./ae4.txt', f"epoch {epoch+1}/{epochs}, train_loss: {loss.item():.6f}")


def load_mnist(batch_size=32, n_train=None, n_test=None):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    if n_train is not None:
        indices = torch.randperm(len(train_full))[:n_train]
        train_full.data = train_full.data[indices]
        train_full.targets = train_full.targets[indices]
    
    if n_test is not None:
        indices = torch.randperm(len(test_full))[:n_test]
        test_full.data = test_full.data[indices]
        test_full.targets = test_full.targets[indices]
    
    train_loader = DataLoader(train_full, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_full, batch_size=batch_size, shuffle=False)
        
    return train_loader, test_loader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train_loader, test_loader = load_mnist(batch_size=32)
    model = AE()
    model.fit(train_loader, test_loader, epochs=50, device=device)
    torch.save(model.state_dict(), './artifacts/ae_last.pth')
    exit()

if __name__ == '__main__':
    main()
