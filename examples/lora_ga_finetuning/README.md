# LoRA-GA: Low-Rank Adaptation with Gradient Approximation

- [LoRA-GA: Low-Rank Adaptation with Gradient Approximation](#lora-ga-low-rank-adaptation-with-gradient-approximation)
  - [introduction paper,code](#introduction-papercode)
  - [preparation](#preparation)
  - [quick start](#quick-start)
  - [What exactly does the above code do?](#what-exactly-does-the-above-code-do)
  - [examples](#examples)
  - [detail usage of functions and classes](#detail-usage-of-functions-and-classes)
    - [LoraGAConfig](#loragaconfig)
    - [estimate\_gradient](#estimate_gradient)
    - [LoraGAContext](#loragacontext)
  - [save\_loraga\_model\_init](#save_loraga_model_init)
  - [save\_loraga\_model\_final](#save_loraga_model_final)
  - [Why do we need to save the model twice?](#why-do-we-need-to-save-the-model-twice)
  - [for quantization model](#for-quantization-model)
    - [lora](#lora)
    - [lora-ga](#lora-ga)
    - [Reason for offload](#reason-for-offload)
    - [offload method](#offload-method)
  - [citation](#citation)

## introduction [paper](https://arxiv.org/abs/2407.05000),[code](https://github.com/Outsider565/LoRA-GA)

[LoRA-GA](https://arxiv.org/abs/2407.05000) aligns the gradients of low-rank matrix product with those of full fine-tuning at the first step. Our extensive experiments demonstrate that LoRA-GA achieves a convergence rate comparable to that of full fine-tuning (hence being significantly faster than vanilla LoRA as well as various recent improvements) while simultaneously attaining comparable or even better performance.

## preparation

```bash
   git clone https://github.com/Outsider565/LoRA-GA.git
   cd LoRA-GA
   git submodule init
   git submodule update peft
   cd peft
   pip install -e .
```

## quick start

```python
from peft import PeftModel, get_peft_model, LoraGAConfig,
from peft.utils.lora_ga_utils import estimate_gradient, LoraGAContext, save_loraga_model_init, save_loraga_model_final

peft_config = LoraGAConfig(
    target_modules=find_all_linear_modules(model=model),
)

named_grad = estimate_gradient(
    model=model,
    dataloader=dataloader,
    accelerator=accelerator,
    quant_flag=False,
)

with LoraGAContext(model=model, named_grad=named_grad):
    model = get_peft_model(model=model, peft_config=peft_config, adapter_name="default")

save_loraga_model_init(model, save_dir=save_dir)

"""
train model
"""

save_loraga_model_final(model, save_dir=save_dir)
# after save_loraga_model_final, you can load it just like you load lora model
PeftModel.from_pretrained(model, save_dir)
```

## What exactly does the above code do?

1. LoraGAConfig is subclass of LoraConfig. LoraGAConfig will set peft_type to PeftType.LORAGA and init_lora_weights = "lora_ga".

2. estimate_gradient will use the data in the dataloader for forward and backward propagation, and return a dictionary named_grad. The key of this dictionary belongs to the submodule name of the model, and the value is the gradient of the weight W of the corresponding module.

3. LoraGAContext will attach named_grad to model as an attribute of model. named_grad will pass named_grad to LoraGAModel which is a subclass of LoraModel. After using named_grad to initialize LoraGAModel(LoraModel), LoraGAModel frees it.

after you get_peft_model, you can use your peft model as lora model to finetune

## examples

1. [example of float model](./float_llama2-7b_metamath.py)

2. [example of quantized model](./quant_llama-2-7b_metamath.py)

## detail usage of functions and classes

### LoraGAConfig

```
@dataclass
class LoraGAConfig(LoraConfig):
    bsz: int = field(
        default=2,
    )
    iters: int = field(
        default=2,
    )
    direction: str = field(
        default="ArB2r",
    )
    max_length: str = field(
        default=1024,
    )
    dtype: str = field(
        default="fp32",
    )
    scale: str = field(default="stable")
    stable_gamma: int = field(
        default=16,
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.LORAGA
        self.init_lora_weights = "lora_ga"
```

dataset of dataloader passed to estimate_gradient should satisfy that:
size of dataset should equal to $bsz * iters$
bsz is batch_size, iters is the number of batches.

### estimate_gradient

```python
def estimate_gradient(
    model,
    dataloader,
    accelerator: Accelerator,
    quant_flag=False,
    origin_type="bf16",
    quant_type="nf4",
    no_split_module_classes=None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Estimate the gradient of the model on the given dataset
    """
```

notice that model should always float model.

if you use float model, you can(should) set quant_flag to Flase to get named_grad faster, but you can also set quant_flag to false to reduce memory overhead(At the same time, increase the time the function runs, because this offload part of model to cpu to make the gpu consumption of the estimated gradient not greater than the gpu memory occupied by the quantized model training later)
if quant_flag is set to False,the three arguments "origin_type, quant_type,no_split_module_classes" will not have any effect.

if you want to pass quantized model to "get_peft_model", you should specify origin_type and quant_type.Currently supported origin types are fp32, bf16 and Currently supported quant types are bitsandbytes int8 and nf4.

no_split_module_classes is used to partition the model. for exmaple it can be residual block, default value is
["LlamaDecoderLayer", "GPT2TransformerBlock", "T5Block", "GPT2Block", "FlaxGPT2Block",]
if you should set no_split_module_classes to your blockname if your model is not in default list.

### LoraGAContext

```python
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
```

LoraGAContext will attach named_grad to model as an attribute of model. named_grad will pass named_grad to LoraGAModel which is a subclass of LoraModel. After using named_grad to initialize LoraGAModel(LoraModel), LoraGAModel free named_grad

## save_loraga_model_init

```python
def save_loraga_model_init(model: PeftModel, save_dir: str):
```

save $A_{init}$ and $B_{init}$

## save_loraga_model_final

```python
def save_loraga_model_final(model: PeftModel, save_dir: str):
```

save $A_{final}$ and $B_{final}$

load $A_{init}$ and $B_{init}$ to init_adapter

load $A_{final}$ and $B_{init}$ to init_adapter

final - init

delete init_adapter

## Why do we need to save the model twice?

when lora-ga initialization is executed, W will be modify:
$$W_{init}=W_{pre\_trained}-\eta B A$$
get $W_{init}, A_{init}, B_{init}$ after LoRA-GA initialization

get $W_{init}, A_{final}, B_{final}$ after the train the peft model

but peft only save weight of adapter, so we need to save $A_{final}-A_{init}$ and $B_{final} - B_{init}$

## for quantization model

for quantized model, estimated_gradient function will spend more time. The reason is below

### lora

$W=W0+\alpha AB$

### lora-ga

lora-ga needs to get the partial derivative of loss with respect to W (equivalent to W0),
but the W0 of the quantized model is stored in integer format,
and pytorch does not support obtaining gradients for integer data.
So please ensure that model passed to estimate_gradient should always float model.

### Reason for offload

In order to get the derivative of loss with respect to W, the original floating-point model
needs to be used to estimate the gradient.

In order to make the gpu consumption of the estimated gradient not greater than the gpu memory occupied by the quantized model training later, it is necessary to offload part of the model to the cpu when estimating the gradient.

### offload method

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

## citation

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
