# Final Assignment - Quantum Federated Learning (QFL)

## Technology and Architecture

Frameworks:

* Flower
* PennyLane
* PyTorch

Libraries:

* flwr[simulation]>=1.28.0
* flwr-datasets[vision]>=0.6.0
* autoray>=0.6,<0.7
* torch
* torchvision
* pennylane

Models 

* Hybird-QCQ CNN for complete CIFAR
* Binary Completelly Quantum CNN for reduced MNIST

Datasets

* CIFAR-10
* Reduced Binary MNIST

For a first experiment, go into `scripts/guilherme_thomaz` from the repository root and run `bash get_repo_example.sh`

Tip for getting Flower port: `sudo ss -tulnp | grep flwr`

## Usage Instructions

Run with two clients:

```bash
flwr run . --stream --federation-config num-supernodes=2
```

Take notes of the run ID. `ex: 15639051639821491333`

Enter `Ctrl+c` to interrupt log stream and move run to background.

To stop the run (`ex: 15639051639821491333`), type:

```bash
flwr stop 15639051639821491333 # Replace the number with your run ID
```

Debug level:

```bash
FLWR_LOG_LEVEL=DEBUG flwr run . --stream --federation-config num-supernodes=2
```

If you had any issues, go into `scripts/guilherme_thomaz` from the repository root and run `bash cleanup.sh`

## Experiments in sequence

```
tsp sh -c "flwr run . --federation-config num-supernodes=4 > run_1.log 2>&1"
tsp sh -c "flwr run . --federation-config num-supernodes=2 > run_1.log 2>&1"
```

## Reference

[Tutorial on Quantum Federated Learning with PennyLane and Flower](https://flower.ai/docs/examples/quickstart-pennylane.html)