"""
    Support Ovis2 Model
"""
import torch
from PIL import Image
from models.base_model import BaseModel
from Constants import *
from transformers import AutoModelForCausalLM

class Ovis2Model(BaseModel):
    def __init__(self, model_path) -> None:
        super().__init__()

        self.core_model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
    
        self.text_tokenizer = self.core_model.get_text_tokenizer()
        self.visual_tokenizer = self.core_model.get_visual_tokenizer()
        self.max_partition = 9

    def __generate_messages(self, text_input, img_filepath) -> tuple[list, Image.Image]:
        image = Image.open(img_filepath).convert("RGB")
        _, input_ids, pixel_values = self.core_model.preprocess_inputs(text_input, [image], max_partition=self.max_partition)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.core_model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.core_model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
        pixel_values = [pixel_values]

        return input_ids, pixel_values, attention_mask

    def infer(self, text_input: str, img_filepath: str, vision_hidden_states=None) -> str:
        input_ids, pixel_values, attention_mask = self.__generate_messages(text_input, img_filepath)

        gen_kwargs = dict(
            max_new_tokens=1,
            do_sample=False,
            top_p=None,
            top_k=None,
            num_beams=1,
            output_scores=True,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.core_model.generation_config.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True
        )

        # Inference: Generation of the output
        output_ids = self.core_model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)
        responses = self.text_tokenizer.decode(output_ids['sequences'][0], skip_special_tokens=True)

        scores = output_ids['scores'][0].squeeze(0)[[INTERNVL2_LOWER_YES, INTERNVL2_LOWER_NO, INTERNVL2_UPPER_YES, INTERNVL2_UPPER_NO]]    
        score = self.compute_score(scores[0], scores[2], scores[1], scores[3])
        
        return responses, None, score