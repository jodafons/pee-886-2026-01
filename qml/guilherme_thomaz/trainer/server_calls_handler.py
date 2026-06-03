from models.CQC import get_initial_cqc_array
from models.BinaryQCNN import get_initial_binqcnn_array 
from flwr.app import ArrayRecord, MetricRecord
from evaluation.eval_engine import global_cqc_evaluate, global_binqcnn_evaluate
from loaders.load_cifar10 import load_cifar10_iid
from loaders.load_mnist import get_mnist 

def get_initial_model_array(context):
    model_name = context.run_config.get("model")
    # TODO: Add more models here as needed
    if model_name == "CQC":
        return get_initial_cqc_array(context)
    elif model_name == "BinaryQCNN":
        return get_initial_binqcnn_array(context)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Create evaluation function with quantum parameters
def make_global_evaluate(context):

    # Load centralized test data (using partition 0 as test set)
    dataset_name = context.run_config.get("dataset")
    test_batch_size = 16
    # TODO: Add more datasets here as needed
    if dataset_name == "cifar10":
        _, testloader = load_cifar10_iid(partition_id=0, num_partitions=1, batch_size=test_batch_size)
    elif dataset_name == "mnist":
        _, testloader = get_mnist(partition_id=0, num_partitions=1, batch_size=test_batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    model_name = context.run_config.get("model")
    # TODO: Add more models here as needed
    if model_name == "CQC":
        def global_evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            return global_cqc_evaluate(server_round, arrays, context, testloader)
        return global_evaluate_fn
    elif model_name == "BinaryQCNN":
        def global_evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            return global_binqcnn_evaluate(server_round, arrays, context, testloader)
        return global_evaluate_fn
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
