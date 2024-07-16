import torch.nn


def get_model_memory(model: torch.nn.Module, forward_factor: float = 1.3):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return forward_factor * total / 1024 ** 3


def get_split_num(origin_type: str = "bf16", quant_type: str = "int8"):
    n_origin_bytes = 16
    n_quant_bytes = 8
    match origin_type:
        case "fp32":
            n_origin_bytes = 32
        case "bf16":
            n_origin_bytes = 16
        case _:
            raise ValueError("Wrong dtype")
    match quant_type:
        case "int8":
            n_quant_bytes = 8
        case "nf4":
            n_quant_bytes = 4
        case _:
            raise ValueError("Wrong dtype")
    return n_origin_bytes // n_quant_bytes
