from typing import Dict, List
from accelerate import Accelerator
import torch
from tqdm import tqdm
import torch.distributed as dist
from peft import LoraGAConfig, LoraConfig


def get_record_gradient_hook(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.cpu()
                else:
                    record_dict[n] += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook


def estimate_gradient(
    model,
    dataloader,
    accelerator: Accelerator,
    quant_flag=False,
    origin_type="bf16",
    quant_type="nf4",
) -> Dict[str, List[torch.Tensor]]:
    r"""
    Estimate the gradient of the model on the given dataset
    """
    if accelerator and model.device.type != "cuda":
        if not quant_flag:
            model.to(accelerator.device)
        else:
            model.to("cpu")
        model.train()
        dataloader = accelerator.prepare(dataloader)
    named_grads = {}
    num_batch = 0
    from offload_utils_for_quant.resource_monitor import show_gpu_and_cpu_memory
    from offload_utils_for_quant.context import OffloadContext

    with OffloadContext(
        model=model,
        named_grads=named_grads,
        quant_flag=quant_flag,
        origin_type=origin_type,
        quant_type=quant_type,
    ):
        for batch in tqdm(dataloader, desc="Estimating gradient"):
            print(f"batch_size=", len(batch["input_ids"]))
            print("before forward===========================================================")
            show_gpu_and_cpu_memory()
            num_batch += 1
            batch = {k: v for k, v in batch.items()}
            outputs = model(**batch)
            show_gpu_and_cpu_memory()
            print("before backward===========================================")
            show_gpu_and_cpu_memory()
            outputs.loss.backward()
            print("after backward ===========================================================")
            show_gpu_and_cpu_memory()

            get_record_gradient_hook(model, named_grads)(None)  # get gradient of last layer
            # make sure the gradient is cleared
            for grad_name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad = None
    for grad_name, _ in named_grads.items():
        named_grads[grad_name] /= num_batch
    torch.cuda.empty_cache()
    if accelerator and accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
        accelerator.print("Gradient estimation finished, gathering results")
        for _, processed_gradient in tqdm(named_grads.items(), desc="Gathering gradient"):
            processed_gradient = processed_gradient.to(accelerator.device)
            dist.all_reduce(processed_gradient, op=dist.ReduceOp.AVG)
            processed_gradient = processed_gradient.to("cpu")
    named_grads = {".".join(k.split(".")[:-1]): v for k, v in named_grads.items()}
    return named_grads


class LoraGAContext:
    def __init__(
        self,
        model: torch.nn.Module,
        named_grad: dict = None,
    ) -> None:
        self.model = model
        self.named_grad = named_grad

    def __enter__(self):
        setattr(self.model, "named_grad", self.named_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.model, "named_grad"):
            delattr(self.model, "named_grad")