import modal
import subprocess
import time

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    # Ставим инструменты сборки и Torch
    .pip_install("setuptools==69.5.1", "packaging", "wheel", "torch==2.1.2")
    # Ставим vLLM отдельно (он потянет 90% нужного для SGLang)
    .pip_install("vllm==0.4.2", "transformers", "hf_transfer")
    # ХАК: Ставим sglang БЕЗ зависимостей, чтобы он не лез за битым flashinfer
    .run_commands(
        "pip install sglang --no-deps",
        "pip install fastapi uvicorn uvloop" # Доставляем то, что реально нужно для сервера
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
    # Мы используем vllm как бэкенд внутри sglang (самый стабильный путь сейчас)
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", "my-secret-swe-bench-token",
        "--mem-fraction-static", "0.8"
    ]

    print(f"Запуск SGLang сервера...")
    process = subprocess.Popen(cmd)
    process.wait()