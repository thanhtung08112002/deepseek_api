pip install fastapi uvicorn transformers torch wandb deepseek

nếu bị lỗi accelerate chạy:
- pip install accelerate>=0.26.0

Chạy API trên Colab hoặc Hugging Face Spaces

pip install colab-tunnel
from colab_tunnel import start_tunnel
start_tunnel(8000)


