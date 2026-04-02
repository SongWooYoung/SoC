#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Qwen3.5 transformers intermediate references for parity work.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-pt", type=Path, default=None)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--max-layers", type=int, default=0, help="Optional cap on dumped decoder layers")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def tensor_stats(tensor: torch.Tensor) -> dict:
    flat = tensor.detach().float().reshape(-1).cpu()
    return {
        "shape": list(tensor.shape),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
    }


def last_token_vector(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor[0, -1, :].detach().float().cpu()
    if tensor.ndim == 4:
        return tensor[0, -1].reshape(-1).detach().float().cpu()
    return tensor.detach().reshape(-1).float().cpu()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prepared_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    input_ids = tokenizer.encode(prepared_prompt, return_tensors="pt")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), dtype=dtype_from_name(args.dtype))
    model = model.to(device)
    model.eval()
    input_ids = input_ids.to(device)

    captures: dict[str, torch.Tensor] = {}
    hooks = []

    def capture(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if isinstance(value, torch.Tensor):
                captures[name] = value.detach()
        return hook

    text_model = model.model
    hooks.append(text_model.embed_tokens.register_forward_hook(capture("embed_tokens")))
    hooks.append(text_model.norm.register_forward_hook(capture("final_norm")))

    layer_limit = len(text_model.layers) if args.max_layers <= 0 else min(args.max_layers, len(text_model.layers))
    for layer_index in range(layer_limit):
        layer = text_model.layers[layer_index]
        hooks.append(layer.input_layernorm.register_forward_hook(capture(f"layer_{layer_index}.input_layernorm")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(capture(f"layer_{layer_index}.post_attention_layernorm")))
        hooks.append(layer.mlp.register_forward_hook(capture(f"layer_{layer_index}.mlp")))
        if hasattr(layer, "self_attn"):
            hooks.append(layer.self_attn.register_forward_hook(capture(f"layer_{layer_index}.token_mixer")))
        if hasattr(layer, "linear_attn"):
            hooks.append(layer.linear_attn.register_forward_hook(capture(f"layer_{layer_index}.token_mixer")))
        hooks.append(layer.register_forward_hook(capture(f"layer_{layer_index}.hidden_out")))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :].detach()

    for hook in hooks:
        hook.remove()

    summary = {
        "prompt": args.prompt,
        "prepared_prompt": prepared_prompt,
        "prompt_token_ids": input_ids[0].detach().cpu().tolist(),
        "next_token_argmax_id": int(logits.argmax(dim=-1).item()),
        "next_token_argmax_text": tokenizer.decode([int(logits.argmax(dim=-1).item())]),
        "captures": {},
    }
    tensors_to_save: dict[str, torch.Tensor] = {
        "logits_last_token": logits[0].detach().float().cpu(),
    }
    for name, tensor in captures.items():
        view = last_token_vector(tensor)
        summary["captures"][name] = tensor_stats(view)
        tensors_to_save[name] = view

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_pt is not None:
        args.output_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors_to_save, args.output_pt)


if __name__ == "__main__":
    main()
