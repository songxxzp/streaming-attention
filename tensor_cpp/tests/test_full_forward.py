#!/usr/bin/env python3
import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer

print("Testing PyTorch full forward pass...")
model = Qwen2ForCausalLM.from_pretrained(
    "/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B",
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained("/media/song/LocalDisk/Storage/checkpoints/Qwen3-0.6B")

# Single token "Hello"
input_ids = torch.tensor([[9707]])

# Forward pass
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

# Get predicted token
predicted_token_id = logits[0, 0, :].argmax().item()
predicted_token = tokenizer.decode([predicted_token_id])

print(f"Input: Hello (token {9707})")
print(f"Predicted token: {predicted_token} (token {predicted_token_id})")
print(f"Logits range: [{logits[0, 0, :].min().item():.2f}, {logits[0, 0, :].max().item():.2f}]")
print(f"Logits mean: {logits[0, 0, :].mean().item():.6f}")
print(f"Top 5 tokens: {logits[0, 0, :].topk(5).indices.tolist()}")
