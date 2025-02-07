import wandb
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

# Đăng nhập vào W&B
wandb.login(key="22e5ffa59d8f9a6efe890ddac8f2f75e5c098971")  

# Khởi tạo W&B và tải dataset
run = wandb.init(project="new-project", name="load-dataset")
artifact = wandb.use_artifact("thanhtungx081102-eyeplus-media/new-project/new-dataset:v0", type="dataset")
artifact_dir = artifact.download()
run.finish()

print(f"Dataset downloaded to: {artifact_dir}")

# Load mô hình DeepSeek
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Khởi tạo FastAPI
app = FastAPI()

@app.post("/chat")
async def chat(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Chạy trên GPU
    with torch.no_grad():
        output = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)

