import os
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import settings
from settings import BASE_DIR
from utils.art import print_logo
from settings import TRAINING_SOURCE_DIR
from config import alpaca_prompt

os.environ["CC"] = "/usr/bin/gcc"



def get_dataset(configs, path="json", split="train"):
    training_json_file = configs["training_source"]
    data_files = os.path.abspath(os.path.join(TRAINING_SOURCE_DIR, training_json_file))

    dataset = load_dataset(
        path=path,
        data_files={
            "train": data_files
        },
        split=split
    )
    return dataset


def set_custom_output_dir(sub_dir):
    from datetime import datetime
    from settings import MODELS_DIR
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.join(MODELS_DIR, f"{sub_dir}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[AI Matrix Finetune] Output Directory: {output_dir}\n")
    return output_dir


def get_model_tokenizer(configs):
    tokenized_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=configs["model_name"],
        max_seq_length=configs["max_seq_length"],
        dtype=torch.float16,
        load_in_4bit=configs["load_in_4bit"],
    )
    print(f"\n[AI Matrix Finetune] Model Tokenized\n")
    return tokenized_model, tokenizer


def formatting_prompts_func(dataset, tokenizer):
    EOS_TOKEN = tokenizer.eos_token

    formatted_texts = []

    for example in dataset:
        text = alpaca_prompt.format(
            example['instruction'],
            example['input'],
            example['output']
        ) + EOS_TOKEN
        formatted_texts.append(text)

    return formatted_texts


def set_peft_model(configs):
    peft_model = FastLanguageModel.get_peft_model(
        configs["tokenized_model"],
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=configs["max_seq_length"],
        use_rslora=False,
        loftq_config=None,
    )
    print(f"\n[AI Matrix Finetune] Pretrained PeftModel Created\n")
    return peft_model


def set_trainer(configs, pretrained_model, tokenizer, dataset):
    training_arguments = TrainingArguments(**configs["TrainingArguments"], output_dir=configs["output_dir"])

    trainer = SFTTrainer(
        model=pretrained_model,
        train_dataset=dataset,
        dataset_text_field=configs.get("dataset_text_field", "text"),
        max_seq_length=configs["max_seq_length"],
        tokenizer=tokenizer,
        args=training_arguments
    )
    print(f"[AI Matrix Finetune] Trainer: \n{trainer}\n")
    return trainer


def perform_inference(pretrained_model, tokenizer):
    # Doesn't it make more sense to do this before and after to check if we screwed up the model?
    FastLanguageModel.for_inference(pretrained_model)
    prompt = "Continue the Fibonacci sequence."
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                prompt, "1, 1, 2, 3, 5, 8", ""  # Example input
            )
        ], return_tensors="pt").to("cuda")
    outputs = pretrained_model.generate(**inputs, max_new_tokens=64, use_cache=True)
    print(tokenizer.batch_decode(outputs))


def save_model_and_tokenizer(configs, pretrained_model, trainer, tokenizer):
    output_dir = configs["output_dir"]
    pretrained_model.save_pretrained(output_dir)
    trainer.save_model(output_dir)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


def show_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")


def fine_tuning_orchestrator(configs):
    dataset = get_dataset(configs, path="json", split="train")

    output_dir = set_custom_output_dir(configs["output_sub_dir"])
    configs['output_dir'] = output_dir

    tokenized_model, tokenizer = get_model_tokenizer(configs)
    configs['tokenized_model'] = tokenized_model

    formatted_texts = formatting_prompts_func(dataset, tokenizer)

    dataset = dataset.map(lambda example, idx: {
        'text': formatted_texts[idx]
    }, with_indices=True)

    pretrained_model = set_peft_model(configs)
    configs['Pretrained peft_model'] = pretrained_model

    print_logo()

    trainer = set_trainer(configs, pretrained_model, tokenizer, dataset)

    trainer_stats = trainer.train()
    print(f"[AI Matrix Finetune] Trainer Stats:\n{trainer_stats}\n")

    perform_inference(pretrained_model, tokenizer)  # It doesn't make sense to me to do interference for pretrained_model. I need to look this up.
    save_model_and_tokenizer(configs, pretrained_model, trainer, tokenizer)
    show_stats()

    print(f"[AI Matrix Finetune] Training Complete!")
    return trainer_stats


def get_training_configs(**kwargs):
    default_configs = {
        "model_name": "unsloth/llama-3-8b-bnb-4bit",
        "training_source": "small_sample.json",
        "output_sub_dir": "unnamed_model",
        "dataset_text_field": "text",
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "TrainingArguments": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 10,
            "max_steps": 100,
            "fp16": True,
            "bf16": False,
            "logging_steps": 1,
            "optim": "adamw_8bit",
            "seed": 3407,
        }
    }

    for key, value in kwargs.items():
        if key in default_configs:
            if isinstance(default_configs[key], dict) and isinstance(value, dict):
                default_configs[key].update(value)
            else:
                default_configs[key] = value

    return default_configs


if __name__ == "__main__":
    new_model_name = "am1a-l38"
    model_name = "unsloth/llama-3-8b-bnb-4bit"
    training_source = "small_sample.json"

    custom_configs = get_training_configs(
        model_name=model_name,
        training_source=training_source,
        output_sub_dir=new_model_name,
        TrainingArguments={
            "max_steps": 10,
            "logging_steps": 1
        }
    )

    trainer_stats = fine_tuning_orchestrator(configs=custom_configs)

