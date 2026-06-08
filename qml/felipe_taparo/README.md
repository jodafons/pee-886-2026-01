# 👤 Student Space

This directory is dedicated to the student's individual implementation. Here you should organize your modules, experiments, and technical documentation.

## 📜 Contribution Rules (Reminder)
1. **Naming Convention**: All files must use `snake_case` (lowercase letters and underscores). Ex: `my_model.py`.
2. **Location**: Work only within the folders identified with your letter (📁 `qml/<letter>/`, 📁 `notebooks/<letter>/`, 📁 `scripts/<letter>/`, 📁 `data/<letter>/`).
3. **Dependencies**: Add new packages only to the `requirements.txt` file at the root of the repository.

## 🛠️ About this Implementation
This work trains an autoencoder and uses it as a feature extractor for QCNNs. The autoencoder is trained on the MNIST dataset. The vector of the latent space of this autoencoder is utilized as input of a QCNN via angle embedding. The QCNN is trained with those inputs while the encoder weights are frozen. Two QCNN architectures for multi-class classification were tested, one with two convolutional layers and no pooling and the other is an ensemble of multiple binary classification QCNN working as a one-vs-all.

### 🚀 Technologies and Architecture
Pennylane was used to simulate quantum circuits.
Torch and Torchvision were used for handling datasets and training models.
Matplotlib and Seaborn were used for plotting.

### 📖 Usage Instructions
Taining the autoencoder:
```bash
python3 autoencoders.py
```

This will train an autoencoder on the MNIST dataset, save the training logs in logs/ae.txt and save the weights in artifacts/ae\_best.txt


For trainig the models:

Autoencoder + QCNN no pool (note ``artifacts/ae_best.pth`` must exist):
```bash
python3 conv.py
```

Autoencoder + hierarchical qcnn:
```bash
python3 hierarchical_ae.py
```

Fully connected + QCNN no pool:
```bash
python3 conv_linear.py
```

The ``plots`` directory has a subdirectory containing logs of previously trained models' test accuracy. Those can be used to generate graphs: 

```bash
python3 plots.py <directory_path>
```

### 📚 References
This works was inspired in the arcticle "T. Hur, L. Kim and D.K. Park, Quantum convolutional neural network for classical data classification, Quantum Machine Intelligence 4 (2022)"

