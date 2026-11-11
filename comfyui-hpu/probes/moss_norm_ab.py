"""Importable A/B switch so Dynamo guards survive runpy's __main__ change."""
from comfy.text_encoders import qwen35

baseline = False
candidate = qwen35.RMSNorm.forward


def forward(module, x):
    return (qwen35.LlamaRMSNorm.forward if baseline else candidate)(module, x)


def install():
    qwen35.RMSNorm.forward = forward
