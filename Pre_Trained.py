from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class PreTrainedModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None
        print("Model loaded")

    def generate_response(self, user_input):
        inputs = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensors="pt"
        )

        bot_input = (
            torch.cat([self.chat_history_ids, inputs], dim=-1)
            if self.chat_history_ids is not None
            else inputs
        )

        self.chat_history_ids = self.model.generate(
            bot_input,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input.shape[-1]:][0],
            skip_special_tokens=True
        ).strip()

        match = re.search(r'^.*?[.!?]', response)
        if match:
            return match.group(0).strip()

        return " ".join(response.split()[:8])
