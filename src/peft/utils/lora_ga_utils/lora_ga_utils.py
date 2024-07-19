from typing import Dict, List
from accelerate import Accelerator
import torch
from tqdm import tqdm
import torch.distributed as dist
from peft import PeftModel


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
    no_split_module_classes=None,
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
    from .offload_utils_for_quant import show_gpu_and_cpu_memory
    from .offload_utils_for_quant import OffloadContext

    with OffloadContext(
        model=model,
        named_grads=named_grads,
        quant_flag=quant_flag,
        origin_type=origin_type,
        quant_type=quant_type,
        no_split_module_classes=no_split_module_classes,
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


def save_peft_model_first(model: PeftModel, save_dir: str):
    import os
    import json

    init_suffix = "_init_lora_checkpoint"
    save_dir = os.path.join(save_dir, init_suffix)
    model.save_pretrained(save_dir)
    adapter_config = json.load(open(os.path.join(save_dir, "adapter_config.json")))
    adapter_config["lora_alpha"] = adapter_config["lora_alpha"]
    print(adapter_config)
    json.dump(adapter_config, open(os.path.join(save_dir, "adapter_config.json"), "w"))


def save_peft_model_second(model: PeftModel, save_dir: str):
    import os
    import json

    final_suffix = "_final_lora_checkpoint"
    save_dir = os.path.join(save_dir, final_suffix)
    model.save_pretrained(save_dir)
    adapter_config = json.load(open(os.path.join(save_dir, "adapter_config.json")))
    adapter_config["lora_alpha"] = adapter_config["lora_alpha"]
    print(adapter_config)
    json.dump(adapter_config, open(os.path.join(save_dir, "adapter_config.json"), "w"))


def load_peft_model(model: torch.nn.Module, save_dir: str, adapter_name=None):
    import os

    init_suffix = "_init_lora_checkpoint"
    final_suffix = "_final_lora_checkpoint"
    merged_suffix = "merged_checkpoint"
    if adapter_name == None:
        adapter_name = "default"
    save_dirs = [f"{save_dir}/{i}" for i in os.listdir(save_dir) if merged_suffix not in i]
    for save_dir in save_dirs:
        dir_name = os.path.split(save_dir)[-1]
        if dir_name == init_suffix:
            tmp_adapter_name = dir_name
        elif dir_name == final_suffix:
            tmp_adapter_name = adapter_name
        else:
            continue
        print(f"loading and from {save_dir}, adapter_name={tmp_adapter_name}")
        model: PeftModel = PeftModel.from_pretrained(model, save_dir, adapter_name=tmp_adapter_name)
        # model = model.merge_and_unload()
    for n, m in model.named_modules():
        if n.endswith("lora_A") or n.endswith("lora_B"):
            m[adapter_name].weight.data -= m[init_suffix].weight.data
        # if n.endswith("lora_A") or n.endswith("lora_B") or n.endswith("lora_dropout"):
        #     m.pop("_init_lora_checkpoint")
        # if isinstance(m, torch.nn.ModuleDict) and init_suffix in m.keys():
        #     m.pop(init_suffix)
    model.base_model.delete_adapter(init_suffix)
    return model


def merge_and_save_lora(model: PeftModel, tokenizer, save_dir: str):
    model = model.merge_and_unload()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    # del model, tokenizer


class LoraGAContext:
    """
    Attach named_grad to the model as an attribute of the model
    """

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
