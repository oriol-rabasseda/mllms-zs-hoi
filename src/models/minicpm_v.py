"""
    Support MiniCPM-V Model
"""
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from typing import List, Optional
from models.base_model import BaseModel
from Constants import *

class MiniCPM_V(BaseModel):
    def __init__(self, model_path) -> None:
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        super().__init__()

        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        dtype = self.config.torch_dtype if hasattr(self.config, "torch_dtype") else torch.float16
        kwargs = {"trust_remote_code": True,
                  "device_map": 'auto',
                  "torch_dtype": dtype,
                  "attn_implementation": "flash_attention_2"}
        if hasattr(self.config, "version") and self.config.version == 2.6:
            kwargs["attn_implementation"] = "sdpa"
            kwargs['device_map'] = 'auto'

        self.core_model = AutoModel.from_pretrained(model_path, **kwargs)
        self.core_model.to(dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.core_model.eval()

    def __generate_V1(
            self,
            data_list=None,
            img_list=None,
            tokenizer=None,
            max_inp_length: Optional[int] = None,
            vision_hidden_states=None,
            return_vision_hidden_states=False,
            **kwargs
    ):
        model_inputs = [None] * len(data_list)
        for i, element in enumerate(data_list):
            model_inputs[i] = self.core_model._process_list(tokenizer, element, max_inp_length)

            pixel_values = []
            img_inps = []
            for img in img_list[i]:
                img_inps.append(self.core_model.transform(img))
            if img_inps:
                pixel_values.append(torch.stack(img_inps).to('cuda'))
            else:
                pixel_values.append([])
            model_inputs[i]['pixel_values'] = pixel_values

            with torch.inference_mode():
                model_inputs[i]['inputs_embeds'], _ = self.core_model.get_vllm_embedding(model_inputs[i])

        input_embeds = torch.cat([mi['inputs_embeds'] for mi in model_inputs], dim=1)

        with torch.inference_mode():
            output = self.core_model.llm.generate(
                inputs_embeds=input_embeds,
                pad_token_id=0,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )

        scores = output['scores'][0].squeeze(0)[[MINICPM_V_LOWER_YES, MINICPM_V_LOWER_NO, MINICPM_V_UPPER_YES, MINICPM_V_UPPER_NO]]
        score = self.compute_score(scores[0], scores[2], scores[1], scores[3])
        result = self.core_model._decode_text(output['sequences'], tokenizer)

        if return_vision_hidden_states:
            return result, vision_hidden_states, score

        else:
            return result, score

    def __generate_V2(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        image_bound=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        decode_text=False,
        **kwargs
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "image_bound": image_bound,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            (
                model_inputs["inputs_embeds"],
                vision_hidden_states,
            ) = self.core_model.get_vllm_embedding(model_inputs)

            output = self.core_model._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, decode_text=False, **kwargs)

            scores = output['scores'][0].squeeze(0)[[INTERNVL2_LOWER_YES, INTERNVL2_LOWER_NO, INTERNVL2_UPPER_YES, INTERNVL2_UPPER_NO]]
            score = self.compute_score(scores[0], scores[2], scores[1], scores[3])

            result = self.core_model._decode_text(output['sequences'], tokenizer)

        if return_vision_hidden_states:
            return result, vision_hidden_states, score

        return result, score.item()


    def __chat_MiniCPM_V_1(self, image, msgs, tokenizer, vision_hidden_states = None, max_new_tokens=1, detected_objects = None):
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        # msgs to prompt
        prompt = ''
        for i, msg in enumerate(msgs):
            role = msg['role']
            content = msg['content']
            if i == 0:
                prompt = [['<用户>' + tokenizer.im_start + tokenizer.unk_token * self.config.query_num + tokenizer.im_end + '\n']]
                if detected_objects != None:
                    for _ in range(len(detected_objects)):
                        prompt += [[tokenizer.im_start + tokenizer.unk_token * self.config.query_num + tokenizer.im_end + '\n']]
                
                prompt[-1][0] += content + '<AI>'
        final_input = prompt

        generation_config = {
            'num_beams': 1,
            'output_scores': True,
            #'output_logits': True,
            'top_k': None,
            'return_dict_in_generate': True
        }

        if detected_objects != None:
            image_list = [[image]] + [[det] for det in detected_objects]
        else:
            image_list = [[image]]

        with torch.inference_mode():
            res, vision_hidden_states_aux, score = self.__generate_V1(
                data_list=final_input,
                max_inp_length=2048,
                img_list=image_list,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                return_vision_hidden_states=True,
                **generation_config
            )
        answer = res[0]

        return answer, vision_hidden_states_aux, score

    def __chat_MiniCPM_V_2(self, image, msgs, tokenizer, vision_hidden_states = None):
        images_list, msgs_list = [image], [msgs]

        self.processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        prompts_lists = []
        input_images_lists = []
        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)

            for i, msg in enumerate(msgs):
                role = msg["role"]
                content = [msg["content"]]
                cur_msgs = ["(<image>./</image>)"]
                for c in content:
                    cur_msgs.append(c)
                msg["content"] = "\n".join(cur_msgs)

            prompts_lists.append(self.processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            input_images_lists.append([image])

        inputs = self.processor(
            prompts_lists,
            input_images_lists,
            return_tensors="pt"
        ).to('cuda')

        generation_config = {
            'num_beams': 1,
            'output_scores': True,
            #'output_logits': True,
            'top_k': None,
            'return_dict_in_generate': True
        }

        inputs.pop("image_sizes")
        with torch.inference_mode():
            res, vision_hidden_states_aux, score = self.__generate_V2(
                **inputs,
                tokenizer=self.tokenizer,
                max_new_tokens=1,
                vision_hidden_states=vision_hidden_states,
                return_vision_hidden_states=True,
                **generation_config
            )

        answer = res[0]

        return answer, vision_hidden_states_aux, score

    def __generate_messages(self, text_input, img_filepath) -> tuple[list, Image.Image]:
        image = Image.open(img_filepath).convert("RGB")
        messages = [{"role": "user", "content": text_input}]

        return messages, image

    def infer(self, text_input: str, img_filepath: str, vision_hidden_states=None, max_new_tokens=1, detected_objects=None):
        messages, image = self.__generate_messages(text_input, img_filepath)

        if hasattr(self.config, "version") and self.config.version in [2.5, 2.6]:
            response, vision_hidden_states, score = self.__chat_MiniCPM_V_2(image=image, msgs=messages,
                                                                            tokenizer=self.tokenizer,
                                                                            vision_hidden_states=vision_hidden_states)

        else:
            response, vision_hidden_states, score = self.__chat_MiniCPM_V_1(image=image, msgs=messages,
                                                                            tokenizer=self.tokenizer,
                                                                            vision_hidden_states=vision_hidden_states,
                                                                            max_new_tokens=max_new_tokens,
                                                                            detected_objects = detected_objects)
        return response, vision_hidden_states, score