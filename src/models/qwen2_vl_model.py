"""
    Support Qwen2-VL Model
"""
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from models.base_model import BaseModel
from Constants import *

class Qwen2VLModel(BaseModel):
    def __init__(self, model_path) -> None:
        super().__init__()

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        dtype = config.torch_dtype if hasattr(config, "torch_dtype") else torch.float16

        if '2.5' in model_path:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.core_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
    
        else:
            from transformers import Qwen2VLForConditionalGeneration
            self.core_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )

        self.core_model.eval()  # Set the model in eval mode explicitly even though not needed
        self.processor = AutoProcessor.from_pretrained(model_path)
    
    def generate_inputs_embeds(self, input_ids, image_grid_thw, pixel_values, image_embeds=None, attention_mask=None):
        inputs_embeds = self.core_model.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.core_model.visual.get_dtype())
            if image_embeds == None:
                image_embeds = self.core_model.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.core_model.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.core_model.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        return inputs_embeds, image_embeds

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

    def infer(self, text_input: str, img_filepath: str, vision_hidden_states=None, max_new_tokens=1, detected_objects=None):
        messages, images = self.__generate_messages(text_input, img_filepath, detected_objects)

        # Preprocess the inputs
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>'
        #   '<|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

        inputs = self.processor(text=[text_prompt], images=images, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")

        #inputs_embeds, image_embeds = self.generate_inputs_embeds(**inputs, image_embeds = vision_hidden_states)
        #inputs["inputs_embeds"] = inputs_embeds.to('cuda:0')
        image_embeds = None

        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        generation_config['num_beams'] = 1
        generation_config['output_scores'] = True
        generation_config['top_k'] = None
        generation_config['return_dict_in_generate'] = True
        generation_config['temperature'] = None
        generation_config['top_p'] = None

        # Inference: Generation of the output
        output_ids = self.core_model.generate(**inputs, **generation_config)

        generated_ids = [
            output_ids['sequences'][len(input_ids):]
            for input_ids, output_ids['sequences'] in zip(inputs.input_ids, output_ids['sequences'])
        ]

        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        scores = output_ids['scores'][0].squeeze(0)[[INTERNVL2_LOWER_YES, INTERNVL2_LOWER_NO, INTERNVL2_UPPER_YES, INTERNVL2_UPPER_NO]]
        score = self.compute_score(scores[0], scores[2], scores[1], scores[3])

        return responses[0], image_embeds, score
