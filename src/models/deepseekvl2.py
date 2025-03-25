"""
    Support DeepSeek-VL2 Model
"""
import torch
from PIL import Image
from models.base_model import BaseModel
from Constants import *
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from transformers import AutoModelForCausalLM

class DeepseekVL2Model(BaseModel):
    def __init__(self, model_path) -> None:
        super().__init__()

        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        self.core_model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path,
                                                                            trust_remote_code=True,
                                                                            device_map='auto',
                                                                            torch_dtype=torch.bfloat16)
    
    def __generate_messages(self, text_input, img_filepath) -> tuple[list, Image.Image]:
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n<|ref|>" + text_input + "<|/ref|>.",
                "images": [img_filepath],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)

        return conversation, pil_images

    def infer(self, text_input: str, img_filepath: str, vision_hidden_states=None) -> str:
        conversation, pil_images = self.__generate_messages(text_input, img_filepath)

        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.core_model.device)
        
        gen_kwargs = dict(
            max_new_tokens=1,
            do_sample=False,
            top_p=None,
            top_k=None,
            num_beams=1,
            output_scores=True,
            temperature=None,
            use_cache=True,
            return_dict_in_generate=True
        )

        # Inference: Generation of the output
        outputs = self.core_model.generate(
            #inputs_embeds=vision_hidden_states,
            input_ids=prepare_inputs.input_ids,
            attention_mask=prepare_inputs.attention_mask,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            labels=prepare_inputs.labels,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs
        )

        generated_ids = [
            outputs['sequences'][len(input_ids):]
            for input_ids, outputs['sequences'] in zip(prepare_inputs.input_ids, outputs['sequences'])
        ]

        responses = self.tokenizer.decode(generated_ids[0].cpu().tolist(), skip_special_tokens=True)

        scores = outputs['scores'][0].squeeze(0)[[DEEPSEEK_LOWER_YES, DEEPSEEK_LOWER_NO, DEEPSEEK_UPPER_YES, DEEPSEEK_UPPER_NO]]    
        score = self.compute_score(scores[0], scores[2], scores[1], scores[3])

        return responses, vision_hidden_states, score