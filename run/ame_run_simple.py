from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import os
import warnings
import logging
import settings
from settings import MODELS_DIR


# Function to load the model and tokenizer
def load_model(model_name):
    model_path = os.path.join(MODELS_DIR, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode
    return tokenizer, model


# Function to generate text with optional streaming
def generate_text(model, tokenizer, user_message, max_length=200, stream=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = tokenizer(user_message, return_tensors='pt').input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Generate text with or without streaming
    with torch.no_grad():
        if stream:
            streamer = TextStreamer(tokenizer, skip_special_tokens=True)
            model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, pad_token_id=tokenizer.eos_token_id, streamer=streamer)
            print()  # Ensure a new line after streaming
        else:
            output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(generated_text)


if __name__ == "__main__":
    model_name = 'l3-8-ama-1a_20240513_1541'

    tokenizer, model = load_model(model_name)

    user_message = "What are some great things I can do with my daughters when we visit New York City?"

    generate_text(model, tokenizer, user_message, stream=True)  # Set stream to True or False as needed
