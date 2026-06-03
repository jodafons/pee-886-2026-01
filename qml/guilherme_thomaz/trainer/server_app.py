import torch
from flwr.app import ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from trainer.server_calls_handler import get_initial_model_array, make_global_evaluate

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    
    # Read run config
    fraction_fit: float = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 1.0)
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    print("Server configuration:")
    print(f"  - Number of rounds: {num_rounds}", flush=True)
    print(f"  - Fraction fit: {fraction_fit}", flush=True)
    print(f"  - Fraction evaluate: {fraction_evaluate}", flush=True)
    print(f"  - Learning rate: {lr}", flush=True)

    arrays = get_initial_model_array(context)

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_train=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=1,
        min_evaluate_nodes=1,
        min_available_nodes=1
    )

    # Start strategy, run FedAvg for `num_rounds`
    print(">>> Antes do strategy.start()", flush=True)
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=make_global_evaluate(context),
    )

    # Save final model to disk
    # print("\nSaving final quantum model to disk...", flush=True)
    # state_dict = result.arrays.to_torch_state_dict()
    # torch.save(state_dict, "final_quantum_model.pt")
