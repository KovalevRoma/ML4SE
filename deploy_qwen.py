import modal
import subprocess
import time

# 1. Настраиваем образ с SGLang и нужными библиотеками
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "sglang[all]", 
        "vllm==0.4.2", 
        "flash-attn",
        "transformers",
        "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}) # Ускоряет скачивание весов
)

app = modal.App("sglang-qwen-serve")

# 2. Создаем постоянное хранилище (Volume), чтобы не качать модель каждый раз
volume = modal.Volume.from_name("model-cache", create_if_missing=True)
MODEL_NAME = "lovedheart/Qwen3.5-4B-FP8"

# 3. Настраиваем деплой функции на GPU
@app.function(
    image=image,
    gpu="A10G", 
    volumes={"/cache": volume},
    timeout=3600,
    min_containers=1, # <--- ВОТ ТУТ ИЗМЕНИЛОСЬ
    allow_concurrent_inputs=10
)
@modal.web_server(8000)
def serve():
    # Запускаем SGLang сервер
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", "my-secret-swe-bench-token", # Твой ключ доступа
        "--mem-fraction-static", "0.8"
    ]

    print(f"Запуск SGLang сервера для {MODEL_NAME}...")
    
    process = subprocess.Popen(cmd)
    
    # Даем серверу немного времени на старт
    time.sleep(30) 
    
    process.wait()