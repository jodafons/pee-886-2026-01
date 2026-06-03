from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from trainer.client_calls_handler import handle_train_call, handle_evaluate_call

app = ClientApp()

@app.train()
def train(msg: Message, context: Context) -> Message:

    model, results = handle_train_call(msg, context)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": results["train_loss"],
        "val_loss": results["val_loss"],
        "val_accuracy": results["val_accuracy"],
        "num-examples": results["num_examples"],
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:

    results = handle_evaluate_call(msg, context)

    # Construct and return reply Message
    metrics = {
        "eval_loss": results["eval_loss"],
        "eval_accuracy": results["eval_accuracy"],
        "custom_metric_1_value": results["custom_metric_1_value"],
        "custom_metric_1_name": results["custom_metric_1_name"],
        "custom_metric_2_value": results["custom_metric_2_value"],
        "custom_metric_2_name": results["custom_metric_2_name"],
        "num-examples": results["num_examples"],
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
