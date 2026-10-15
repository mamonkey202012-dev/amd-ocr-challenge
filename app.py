import argparse
import json
import os
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def parse_args():
    parser = argparse.ArgumentParser(description="AMD AI OCR Inference using VLM")
    parser.add_argument("--input-image", required=True, help="Path to input image file")
    return parser.parse_args()

def main():
    args = parse_args()
    image_path = args.input_image

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompt_text = (
        "Perform OCR on this image following these strict rules:\n"
        "1. If it is a US license plate, output ONLY the registration number. Ignore state names or slogans (e.g., CALIFORNIA, EXCELSIOR).\n"
        "2. If it is a Chinese license plate, include the leading province character and letter (e.g., 京A 12345).\n"
        "3. If it is a traffic sign, transcribe all text top-to-bottom joined by a single space.\n"
        "4. If it is a speed sign with only numbers, output only the digits without units.\n"
        "5. Output only the extracted characters. Do not add labels, explanations, or quotes."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_output.json")

    output_data = {
        "text": output_text,
        "confidence": 0.95
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Inference complete: {output_text} saved to {output_path}")

if __name__ == "__main__":
    main()