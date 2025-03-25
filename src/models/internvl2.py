"""
    Support InternVL2 Model
"""
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from models.internvl2_utils import *
import importlib
from PIL import Image
from models.base_model import BaseModel
from Constants import *
import math

class InternVL2Model(BaseModel):
    def __init__(self, model_path) -> None:
        super().__init__()
        #config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        #num_hidden_layers = config.llm_config.num_hidden_layers
        #device_map = split_model(num_hidden_layers)

        self.core_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map='auto',
            ).eval()
        #self.core_model.to(device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def __generate_messages(self, text_input, img_filepath):
        image = Image.open(img_filepath).convert("RGB")
        pixel_values = load_image(image).to(device='cuda', dtype=torch.bfloat16)
        question = '<image>\n' + text_input

        libname = '.'.join(str(type(self.core_model)).split("'")[1].split('.')[:4] + ['conversation'])
        conversation = importlib.import_module(libname)

        num_patches_list = [pixel_values.shape[0]]

        img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.core_model.img_context_token_id = img_context_token_id

        template = conversation.get_conv_template(self.core_model.template)
        template.system_message = self.core_model.system_message

        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = '<img>' + '<IMG_CONTEXT>' * self.core_model.num_image_token * num_patches + '</img>'
            query = query.replace('<image>', image_tokens, 1)

        return query, pixel_values

    def infer(self, text_input: str, img_filepath: str, vision_hidden_states=None):
        query, pixel_values = self.__generate_messages(text_input, img_filepath)

        generation_config = dict(max_new_tokens=1, do_sample=False)
        model_inputs = self.tokenizer(query, return_tensors='pt')
        generation_config['eos_token_id'] = 51645
        generation_config['pad_token_id'] = 51645
        generation_config['num_beams'] = 1
        generation_config['output_scores'] = True
        generation_config['top_k'] = None
        generation_config['return_dict_in_generate'] = True

        generation_output = self.core_model.generate(
            pixel_values=pixel_values,
            input_ids=model_inputs['input_ids'].to(device='cuda'),
            attention_mask=model_inputs['attention_mask'].to(device='cuda'),
            **generation_config
        )

        responses = self.tokenizer.batch_decode(generation_output['sequences'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        scores = generation_output['scores'][0].squeeze(0)[[INTERNVL2_LOWER_YES, INTERNVL2_LOWER_NO, INTERNVL2_UPPER_YES, INTERNVL2_UPPER_NO]]
        score = self.compute_score(scores[0], scores[2], scores[1], scores[3])

        return responses[0], None, score
