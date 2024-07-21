import torch

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from peft import PeftModel, LoraGAConfig, get_peft_model
from peft.utils.lora_ga_utils import estimate_gradient, LoraGAContext, save_loraga_model_init, save_loraga_model_final

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
from accelerate import Accelerator
from utils import transform_dataset, initialize_text_to_text_model, find_all_linear_modules
from data import DATASET_MAP


def main():
    accelerator = Accelerator()
    model_id = "meta-llama/Llama-2-7b-hf"
    model_type = "CausalLM"
    model_dtype = "bf16"
    model, tokenizer = initialize_text_to_text_model(model_id, model_type, model_dtype, flash_attention=True)
    print(model)
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
    print(peft_config)
    with LoraGAContext(model=model, named_grad=named_grad):
        model = get_peft_model(model=model, peft_config=peft_config)
    save_dir = "snapshot"
    save_loraga_model_init(model=model, save_dir=save_dir)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print("finish get_peft_model=================================================")
    """
    train peft model
    """
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    save_loraga_model_final(model=model, save_dir=save_dir)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    model, tokenizer = initialize_text_to_text_model(model_id, model_type, model_dtype, flash_attention=True)
    model = PeftModel.from_pretrained(model, save_dir)
    print(model)


if __name__ == "__main__":
    main()
