"""
    Support Phi-Vision Model
"""
import torch
from PIL import Image
from models.base_model import BaseModel
from Constants import *
from transformers import AutoModelForCausalLM, AutoProcessor

class PhiVisionModel(BaseModel):
    def __init__(self, model_path) -> None:
        super().__init__()

        self.core_model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            device_map='auto',
                            trust_remote_code=True,
                            torch_dtype="auto",
                            _attn_implementation='flash_attention_2'
        )

        self.processor = AutoProcessor.from_pretrained(model_path,
                                                  trust_remote_code=True,
                                                  )
        self.core_model.eval()  # Set the model in eval mode explicitly even though not needed
    

    def __generate_messages(self, text_input, img_filepath) -> tuple[list, Image.Image]:
        image = Image.open(img_filepath).convert("RGB")
        messages = [
            {"role": "user", "content": f"<|image_1|>\n" + text_input},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt, image

    def infer(self, text_input: str, img_filepath: str, vision_hidden_states=None) -> str:
        messages, image = self.__generate_messages(text_input, img_filepath)

        inputs = self.processor(messages, [image]).to('cuda')

        generation_config = dict(max_new_tokens=1, do_sample=False)
        generation_config['num_beams'] = 1
        generation_config['output_scores'] = True
        generation_config['top_k'] = None
        generation_config['return_dict_in_generate'] = True
        generation_config['eos_token_id'] = self.processor.tokenizer.eos_token_id


        # Inference: Generation of the output
        output_ids = self.core_model.generate(**inputs, **generation_config)
        
        generated_ids = [
            output_ids['sequences'][len(input_ids):]
            for input_ids, output_ids['sequences'] in zip(inputs.input_ids, output_ids['sequences'])
        ]

        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        scores = output_ids['scores'][0].squeeze(0)[[PHIVISION_LOWER_YES, PHIVISION_LOWER_NO, PHIVISION_UPPER_YES, PHIVISION_UPPER_NO]]    
        score = self.compute_score(scores[0], scores[2], scores[1], scores[3])
        
        return responses[0], None, score