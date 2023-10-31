import torch


# Default MACE optimizer params
def get_default_optimizer(
    learning_rate: float,
    model_wrapper: torch.nn.Module,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
):
    decay_interactions = {}
    no_decay_interactions = {}
    for name, param in model_wrapper.model.interactions.named_parameters():
        if "linear.weight" in name or "skip_tp_full.weight" in name:
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    param_options = dict(
        params=[
            {
                "name": "embedding",
                "params": model_wrapper.model.node_embedding.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "noise_linear",
                "params": model_wrapper.model.noise_linear.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "interactions_decay",
                "params": list(decay_interactions.values()),
                "weight_decay": weight_decay,
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model_wrapper.model.products.parameters(),
                "weight_decay": weight_decay,
            },
            {
                "name": "readouts",
                "params": model_wrapper.model.readouts.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        amsgrad=amsgrad,
    )
    optimizer = torch.optim.AdamW(**param_options)
    return optimizer
