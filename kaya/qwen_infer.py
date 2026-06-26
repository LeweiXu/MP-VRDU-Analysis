#!/usr/bin/env python
"""Minimal Qwen inference smoke test for Kaya.

Text-only:
    python qwen_infer.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "Say hi in one sentence."

Vision-language (pass an image, requires qwen-vl-utils):
    python qwen_infer.py --model Qwen/Qwen2.5-VL-3B-Instruct \
        --prompt "Describe this image." --image path/to/image.png
"""
import argparse

import torch


def run_text(model_id: str, prompt: str, max_new_tokens: int) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_vision(model_id: str, prompt: str, image_path: str, max_new_tokens: int) -> str:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default="Say hi in one sentence.")
    parser.add_argument("--image", default=None, help="path to an image; switches to the VL path")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading {args.model} ...")

    if args.image:
        reply = run_vision(args.model, args.prompt, args.image, args.max_new_tokens)
    else:
        reply = run_text(args.model, args.prompt, args.max_new_tokens)

    print("--- reply ---")
    print(reply)


if __name__ == "__main__":
    main()
