#!/bin/bash
pip install --upgrade typer flwr
flwr new @flwrlabs/quickstart-pennylane
cd quickstart-pennylane
pip install -e .
flwr run .  --stream
cd ..
rm -rf quickstart-pennylane