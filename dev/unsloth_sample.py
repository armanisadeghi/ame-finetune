import os

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import settings
from settings import BASE_DIR
from settings import MODELS_DIR
from settings import TRAINING_SOURCE_DIR
from config import alpaca_prompt

# TODO:  Need to fully understand what this is and what the implications are.
max_seq_length = 2048

print(f'BASE_DIR: {BASE_DIR}')
print(f'MODELS_DIR: {MODELS_DIR}')
print(f'TRAINING_SOURCE_DIR: {TRAINING_SOURCE_DIR}')


# Construct the full path to the small_sample.json file
training_json_file = os.path.abspath(os.path.join(TRAINING_SOURCE_DIR, "small_sample.json"))
output_dir = MODELS_DIR

print(f"training_json_file: {training_json_file}")
print(f"output_dir: {output_dir}")


# Make sure the file exists
if not os.path.exists(training_json_file):
    raise FileNotFoundError(f"Unable to find '{training_json_file}'")

print(f"Using dataset file: {training_json_file}")

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# More models at https://huggingface.co/unsloth

# TODO:  While 4bit is good for fast training, these models are not anywhere near as good in terms of quality
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
)


EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
        }


pass

dataset = load_dataset(
    path="json",
    data_files={
        "train": training_json_file
        },
    split="train")
dataset = dataset.map(formatting_prompts_func, batched=True, )

# Do model patching and add fast LoRA weights
# TODO: Consider qLora for faster training or on smaller GPUs - Need to research the implications of it.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    max_seq_length=max_seq_length,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=1,  # Used 2 for first test
        gradient_accumulation_steps=4,
        warmup_steps=10,  # Used 500 for first test
        max_steps=60,  # Used 10,000 for first test (But It might be too much for simple data) # TODO: Need to do testing and see what makes the most sense.
        fp16=True,
        bf16=False,
        logging_steps=1,  # Used 100 for first test and it's good. If being watched closely for a short test, more is ok, but not necessary at all.
        output_dir=output_dir,  # Updated to go into the MODELS_DIR so it's organized and out of git.
        optim="adamw_8bit",
        seed=3407,
    ),
)

# Show current memory stats

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer.train()

# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Continue the fibonnaci sequence.",  # instruction
            "1, 1, 2, 3, 5, 8",  # input
            "",  # output - leave this blank for generation!
        )
    ], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
tokenizer.batch_decode(outputs)

# Save the trained model
model.save_pretrained(output_dir)
trainer.save_model(output_dir)
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
