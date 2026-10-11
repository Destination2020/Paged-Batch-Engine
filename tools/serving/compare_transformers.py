#!/usr/bin/env python3
"""Compare the validation prompts with local Transformers BF16 eager inference.

Returns 2 on text differences; this is a diagnostic, not a logits tolerance test.
Run in the Python environment containing torch and transformers.
"""
import argparse
import gc
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text())
    validation = json.loads(args.validation.read_text())
    prompts = ["请用一句话介绍中国的首都。", "Write a short Python function to add two numbers.",
               "请总结以下内容：" + "今天的会议讨论了共享显存和并发推理。" * 24]
    results = {}
    for entry in config["models"]:
        weights = Path(entry["weights"])
        directory = (weights if weights.is_absolute() else root / weights).parent
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            directory, dtype=torch.bfloat16, local_files_only=True,
            attn_implementation="eager").to("cuda").eval()
        outputs = []
        for prompt in prompts:
            # Match demo/serving_qwen.cpp's explicit default template, including
            # the newline before each im_end marker.
            text = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
                    "You are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n" +
                    prompt + "\n<|im_end|>\n<|im_start|>assistant\n")
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=24, do_sample=False)
            outputs.append(tokenizer.decode(tokens[0, inputs["input_ids"].shape[1]:],
                                            skip_special_tokens=True).rstrip())
        baseline = validation["outputs"][entry["id"]]
        if len(baseline) != len(prompts):
            raise ValueError("Validation report does not contain the expected three prompts")
        results[entry["id"]] = dict(hf_outputs=outputs, pbe_outputs=baseline,
                                    text_match=[a == b for a, b in zip(outputs, baseline)])
        print(json.dumps({entry["id"]: results[entry["id"]]}, ensure_ascii=False), flush=True)
        del model, inputs, tokens
        gc.collect()
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(all(result["text_match"]) for result in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
