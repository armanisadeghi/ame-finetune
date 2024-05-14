from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import os
import warnings
import logging
import settings
from settings import MODELS_DIR


class AmeTextStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_special_tokens=True)
        self.generated_text = ""
        self.stream_end = False

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Collects the generated text and prints it to stdout."""
        self.generated_text += text
        if stream_end:
            self.stream_end = True
        super().on_finalized_text(text, stream_end)


class Matrix:
    def __init__(self):
        self.model_list = os.listdir(MODELS_DIR)
        self.models = []

    def list(self):
        print(f"\n[AI Matrix] Custom Models:")
        for model_name in self.model_list:
            print(f" - {model_name}")
        print()
        return self.model_list

    def load(self, model_name):
        model = AmaModel(model_name)
        return model

    def load_all_models(self):
        for model_name in self.model_list:
            model = AmaModel(model_name)
            self.models.append(model)
        return self.models


class AmaModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer, self.model = self.load(model_name)

    def load(self, model_name):
        model_path = os.path.join(MODELS_DIR, model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        return tokenizer, model

    def chat(self, input_text, max_length=200, stream=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids.to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        generated_text = ""

        with torch.no_grad():
            if stream:
                custom_streamer = AmeTextStreamer(self.tokenizer)
                self.model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id, streamer=custom_streamer)
                while not custom_streamer.stream_end:
                    pass
                generated_text = custom_streamer.generated_text
            else:
                output_ids = self.model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generated_text


if __name__ == "__main__":
    matrix = Matrix()
    model_list = matrix.list()

    model_name = 'l3-8-ama-1a_20240513_1541'
    model = matrix.load(model_name)

    SINGLE_QUESTION = False

    if SINGLE_QUESTION:
        user_message = "What are some great things I can do with my daughters when we visit New York City?"
        response = model.chat(user_message, stream=True)
        print(f"\nFull Response:\n{response}\n")
        exit()

    else:
        while True:
            user_message = input("You: ")
            if user_message.lower() in ['exit', 'quit']:
                break
            response = model.chat(user_message, stream=True)
            print(f"AI Matrix:\n{response}\n")
