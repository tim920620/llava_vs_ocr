import os
import torch
import easyocr

image_path = "data/images/26d9378d1d055421.jpg"

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("image exists:", os.path.exists(image_path))
print("image path:", image_path)

reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
result = reader.readtext(image_path, detail=0)

print("OCR result:", result)
print("OCR text:", " ".join(result))