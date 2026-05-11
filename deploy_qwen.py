import modal
import subprocess
import time

# 1. Настраиваем образ: используем правильные параметры для новых версий Modal
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("torch==2.1.2") 
    .pip_install(
        "sglang[all]", 
        "vllm==0.4.2", 
        "flash-attn",
        "transformers",
        "hf_transfer",
        extra_options="--no-build-isolation" # Теперь это в правильном месте
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("sglang-qwen-serve")

volume = modal.Volume.from_name("model-cache", create_if_missing=True)
MODEL_NAME = "lovedheart/Qwen3.5-4B-FP8"

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": volume},
    timeout=3600,
    min_containers=1
)
@modal.web_server(8000)
def serve():
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", "my-secret-swe-bench-token",
        "--mem-fraction-static", "0.8"
    ]

    print(f"Запуск SGLang сервера для {MODEL_NAME}...")
    process = subprocess.Popen(cmd)
    time.sleep(30) 
    process.wait()