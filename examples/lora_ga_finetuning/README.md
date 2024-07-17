# LoRA-GA: Low-Rank Adaptation with Gradient Approximation

- [LoRA-GA: Low-Rank Adaptation with Gradient Approximation](#lora-ga-low-rank-adaptation-with-gradient-approximation)
  - [introduction paper,code](#introduction-papercode)
  - [quick start](#quick-start)
  - [for quantized model](#for-quantized-model)
  - [Citation](#citation)

## introduction [paper](https://arxiv.org/abs/2407.05000),[code](https://github.com/Outsider565/LoRA-GA)

[LoRA-GA](https://arxiv.org/abs/2407.05000) aligns the gradients of low-rank matrix product with those of full fine-tuning at the first step. Our extensive experiments demonstrate that LoRA-GA achieves a convergence rate comparable to that of full fine-tuning (hence being significantly faster than vanilla LoRA as well as various recent improvements) while simultaneously attaining comparable or even better performance.

## quick start

Configure the initialization method to "lora_ga" by using LoraGAConfig, then get estimated_grad, and get peft model with LoraGAContext:

```python
import torch
from peft import LoraGAConfig, get_peft_model
from peft.utils.lora_ga_utils import estimate_gradient, LoraGAContext
from accelerate import Accelerator
from utils import transform_dataset, initialize_text_to_text_model, find_all_linear_modules
from data import DATASET_MAP


def main():
    accelerator = Accelerator()
    model_id = "meta-llama/Llama-2-7b-hf"
    model_type = "CausalLM"
    model_dtype = "bf16"
    model, tokenizer = initialize_text_to_text_model(model_id, model_type, model_dtype, flash_attention=True)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    peft_config = LoraGAConfig(
        target_modules=find_all_linear_modules(model=model),
    )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    dataset_name = "meta_math"
    dataset_func = DATASET_MAP[dataset_name]
    train_set, val_set, _ = dataset_func()
    if isinstance(train_set, list):
        temp_set = train_set[: peft_config.bsz * peft_config.iters]
    else:
        temp_set = train_set.select(range(peft_config.bsz * peft_config.iters))
    transform_dataset(
        model_type=model_type,
        dataset=temp_set,
        tokenizer=tokenizer,
        max_length=peft_config.max_length,
    )
    dataloader = torch.utils.data.DataLoader(temp_set, batch_size=peft_config.bsz)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    named_grad = estimate_gradient(
        model=model,
        dataloader=dataloader,
        accelerator=accelerator,
        quant_flag=False,
    )
    with LoraGAContext(model=model, named_grad=named_grad):
        print(peft_config)
        model = get_peft_model(model=model, peft_config=peft_config, adapter_name="default")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(model)


if __name__ == "__main__":
    main()
```

## for quantized model

```python
import torch
from peft import LoraGAConfig, get_peft_model
from peft.utils.lora_ga_utils import estimate_gradient, LoraGAContext
from accelerate import Accelerator
from utils import (
    transform_dataset,
    initialize_text_to_text_model,
    find_all_linear_modules,
)
from data import DATASET_MAP


def main():
    accelerator = Accelerator()
    model_id = "meta-llama/Llama-2-7b-hf"
    model_type = "CausalLM"
    # model_type should be float before estimate_grad
    model_dtype = "bf16"
    model, tokenizer = initialize_text_to_text_model(model_id, model_type, model_dtype, flash_attention=False)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    peft_config = LoraGAConfig(
        target_modules=find_all_linear_modules(model=model),
    )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    dataset_name = "meta_math"
    dataset_func = DATASET_MAP[dataset_name]
    train_set, val_set, _ = dataset_func()
    if isinstance(train_set, list):
        temp_set = train_set[: peft_config.bsz * peft_config.iters]
    else:
        temp_set = train_set.select(range(peft_config.bsz * peft_config.iters))
    transform_dataset(
        model_type=model_type,
        dataset=temp_set,
        tokenizer=tokenizer,
        max_length=peft_config.max_length,
    )
    dataloader = torch.utils.data.DataLoader(temp_set, batch_size=peft_config.bsz)
    print(f"len(dataloader)={len(dataloader)}")
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """
    re-get the quant-model
    """
    quant_type = "nf4"
    named_grad = estimate_gradient(
        model=model,
        dataloader=dataloader,
        accelerator=accelerator,
        quant_flag=True, # if you have GPU memory enough, you can also set quant_flag=Ture to acclerate estimate_gradient
        origin_type="bf16",
        quant_type=quant_type,
        no_split_module_classes=["your ","block","name"]
        """
        no_split_module_classes defualt is ["LlamaDecoderLayer", "GPT2TransformerBlock", "T5Block", "GPT2Block","FlaxGPT2Block",]
        no_split_module_classes will be used to split model (to k block) for offload.
        """
    )
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    model_dtype = quant_type
    model, tokenizer = initialize_text_to_text_model(model_id, model_type, model_dtype, flash_attention=False)
    # ++++++++++++++++++++++++++++++++++++++++++++++++

    with LoraGAContext(model=model, named_grad=named_grad):
        print(peft_config)
        model = get_peft_model(model=model, peft_config=peft_config, adapter_name="default")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(model)

if __name__ == "__main__":
    main()

```

lora-ga needs to get the partial derivative of loss with respect to W (equivalent to W0),
but the W0 of the quantized model is stored in integer format,
and pytorch does not support obtaining gradients for integer data.

1. In order to get the derivative of loss with respect to W, the original floating-point model
   needs to be used to estimate the gradient.

2. In order to make the gpu consumption of the estimated gradient not greater than the gpu memory occupied by
   the quantized model training later,
   it is necessary to offload part of the model to the cpu when estimating the gradient.

model offload method:

```pytoon
Initial: Model on the cpu

Divide the model into K blocks.
for i in range(0,K):
    Load the i-th block to the gpu
    Execute the forward of the i-th block
    if i != K-1:
        # If it is not the last block, offload the i-th block to the cpu
        offload the i-th block to the cpu
    else：
        # If it is the last block, because the last block needs to be back-propagated first, no offload is needed
        do nothing
for i in range(K-1, -1, -1):
    Load the i-th block to the gpu
    Execute the backward of the i-th block
    if i != 0:
        # If it is not the 0th block, offload the i-th block to the cpu
        offload the i-th block to the cpu
    else：
        # If it is block 0, since the forward of the next batch first requires block 0 to be forwarded, no offload
        do nothing
```

## Citation

```
@misc{wang2024loragalowrankadaptationgradient,
      title={LoRA-GA: Low-Rank Adaptation with Gradient Approximation},
      author={Shaowen Wang and Linxi Yu and Jian Li},
      year={2024},
      eprint={2407.05000},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.05000},
}
```
