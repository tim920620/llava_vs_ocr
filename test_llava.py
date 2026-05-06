import os
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"
image_path = "data/images/26d9378d1d055421.jpg"

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("image exists:", os.path.exists(image_path))
print("image path:", image_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("device:", device)
print("dtype:", dtype)

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
).to(device)

model.eval()

image = Image.open(image_path).convert("RGB")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Read all visible text in the image. List every word or phrase you can find."},
            {"type": "image"}
        ]
    }
]

prompt = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True
)

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
)

inputs = {
    k: v.to(device) if hasattr(v, "to") else v
    for k, v in inputs.items()
}

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False
    )

input_token_len = inputs["input_ids"].shape[1]
generated_ids = output[:, input_token_len:]

answer = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)[0]

print("LLaVA answer:", answer.strip())