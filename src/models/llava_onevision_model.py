"""
Support LLaVA-OneVision Huggingface model
"""
import torch
from PIL import Image
from transformers import AutoConfig
from models.base_model import BaseModel
from Constants import *


class LLaVAOneVisionModel(BaseModel):
    def __init__(self, model_path) -> None:
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

        super().__init__()

        self.core_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='cuda:2',
            attn_implementation='flash_attention_2',
            #max_memory={0: "5GB", 1: "8GB", 2: "8GB"}
            )
        self.core_model.eval()  # Set the model in eval mode explicitly even though not needed
        self.processor = AutoProcessor.from_pretrained(model_path)

    def __generate_messages(self, text_input, img_filepath, detected_objects=None) -> tuple[list, Image.Image]:
        messages = []
        content = []
        image = Image.open(img_filepath).convert("RGB")

        if detected_objects != None:
            new_detected_objects = [det for det in detected_objects if det.size[0] > 28 and det.size[1] > 28]
            images = new_detected_objects + [image]
            content.append({"type": "text", "text": "Relevant objects in the image:"})
            content += [{"type": "image"}] * len(new_detected_objects)
            content.append({"type": "text", "text": "Original image:"})
        else:
            images = [image]

        content.append({"type": "image"})
        content.append({"type": "text", "text": text_input})
        messages.append({"role": "user", "content": content})

        return messages, images

    def infer(self, text_input: str, img_filepath: str, vision_hidden_states=None, detected_objects=None) -> str:
        messages, images = self.__generate_messages(text_input, img_filepath, detected_objects)

        # Preprocess the inputs
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(images=images, text=text_prompt, return_tensors='pt').to('cuda:2', torch.float16)

        generation_config = dict(max_new_tokens=1, do_sample=False)
        generation_config['num_beams'] = 1
        generation_config['output_scores'] = True
        generation_config['top_k'] = None
        generation_config['return_dict_in_generate'] = True
        generation_config['pad_token_id'] = 51645

        output_ids = self.core_model.generate(**inputs, **generation_config)
        generated_ids = output_ids['sequences'][0][len(inputs.input_ids[0]):]

        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        scores = output_ids['scores'][0].squeeze(0)[[INTERNVL2_LOWER_YES, INTERNVL2_LOWER_NO, INTERNVL2_UPPER_YES, INTERNVL2_UPPER_NO]]
        score = self.compute_score(scores[0], scores[2], scores[1], scores[3])

        return responses[0], None, score
