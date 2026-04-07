import os


def resolve_qwen_dtype(torch_module):
    override = os.getenv("SOFAR_QWEN_DTYPE", "").strip().lower()
    dtype_map = {
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }
    if override in dtype_map:
        return dtype_map[override]

    if torch_module.cuda.is_available():
        major, _ = torch_module.cuda.get_device_capability()
        if major >= 8:
            return torch_module.bfloat16
        return torch_module.float16

    return torch_module.float32
