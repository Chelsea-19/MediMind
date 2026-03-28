import torch
import json
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

class MultimodalMedicalModel:
    def __init__(self, model_name: str, quantization: str = "4bit", device_map: str = "auto"):
        logger.info(f"Loading Generative Model: {model_name} (quantization={quantization})")
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        # Load Model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def generate_response(self, system_prompt: str, user_text: str, image_path: Optional[str] = None, json_mode: bool = True) -> str:
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        content = []
        if image_path:
            content.append({"type": "image", "image": image_path})
            
        content.append({"type": "text", "text": user_text})
        
        if json_mode:
            content.append({"type": "text", "text": "务必以要求的 JSON 格式输出。"})
            
        messages.append({"role": "user", "content": content})
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to same device as model
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)
        else:
            inputs = inputs.to("cuda")

        try:
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return output
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return '{"error": "inference failed"}'
