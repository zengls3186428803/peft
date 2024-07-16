import torch.nn
from .model_offload import ModelOffloadHookContext
from .split import get_split_num
from .gradient_offload import GradientOffloadHookContext



class OffloadContext:
    def __init__(
            self,
            model: torch.nn.Module,
            named_grads: dict,
            quant_flag: bool = False,
            origin_type: str = "bf16",
            quant_type: str = "int8",
            no_split_module_classes=None,
            enable_gradient_offload=True,
    ):
        if no_split_module_classes is None:
            no_split_module_classes = [
                "LlamaDecoderLayer", "GPT2TransformerBlock", "T5Block", "GPT2Block", "FlaxGPT2Block",
            ]
        num_split_block = get_split_num(origin_type=origin_type, quant_type=quant_type)
        if quant_flag:
            print(f"model will be split into {num_split_block} blocks")

        self.modelOffloadHookContext = ModelOffloadHookContext(
            model=model,
            no_split_module_classes=no_split_module_classes,
            num_block=num_split_block,
            enable=quant_flag,
            # =========================
            device="cuda",
            strategy="block",
            with_backward_hook=False
        )
        self.gradientOffloadHookContext = GradientOffloadHookContext(
            model=model,
            enable=enable_gradient_offload,
            record_dict=named_grads,
        )

    def __enter__(self):
        self.modelOffloadHookContext.__enter__()
        self.gradientOffloadHookContext.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.modelOffloadHookContext.__exit__(exc_type, exc_val, exc_tb)
        self.gradientOffloadHookContext.__exit__(exc_type, exc_val, exc_tb)
