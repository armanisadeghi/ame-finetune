# Project Root /config.py
# This is a central place to store all the configuration settings for the project.
# Avoid having configuration settings scattered throughout the codebase, unless something is highly specific to a single script.

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that
provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

AVAILABLE_UNSLOTH_MODELS = {
    "fourbit_models": [
        "unsloth/tinyllama-bnb-4bit"
        "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
        "unsloth/llama-2-13b-bnb-4bit"
        "unsloth/zephyr-sft-bnb-4bit"
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
        "unsloth/llama-3-8b-bnb-4bit",  # [NEW] 15 Trillion token Llama-3
        "unsloth/codegemma-7b-bnb-4bit"
        "unsloth/gemma-2b-bnb-4bit"
        "unsloth/mistral-7b-bnb-4bit"
        "unsloth/llama-2-7b-bnb-4bit"
        "unsloth/yi-6b-bnb-4bit"
        "unsloth/solar-10.7b-bnb-4bit"
        "unsloth/llama-3-70b-bnb-4bit"
        "unsloth/mistral-7b-v0.2-bnb-4bit"
        "unsloth/codellama-34b-bnb-4bit"
        "unsloth/gemma-7b-bnb-4bit"
        "unsloth/codellama-7b-bnb-4bit"
    ],
    "fourbit_instruct_models": [
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "unsloth/llama-3-70b-Instruct-bnb-4bit",
        "unsloth/gemma-7b-it-bnb-4bit",  # Instruct version of Gemma 7b
        "unsloth/gemma-2b-it-bnb-4bit",  # Instruct version of Gemma 2b
        "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit",
        "unsloth/Hermes-2-Pro-Mistral-7B-bnb-4bit",
    ]
}




