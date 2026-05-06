import json
import os
print("test")
INPUT_JSON = "data/textvqa/TextVQA_0.5.1_val.json"
IMAGE_DIR = "data/images"
OUTPUT_JSON = "data/samples.json"

MAX_SAMPLES = 300  # 先小量測試，之後可改大


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    count = 0

    # TextVQA 常見格式: data["data"] 裡面是一筆一筆樣本
    items = data["data"]

    for item in items:
        image_id = item["image_id"]
        question = item["question"]
        answers = item.get("answers", [])

        if not answers:
            continue

        # 多數情況直接取第一個答案
        gold_answer = answers[0]

        # TextVQA 圖片通常是 jpg
        image_filename = f"{image_id}.jpg"
        image_path = os.path.join(IMAGE_DIR, image_filename)

        if not os.path.exists(image_path):
            # 有些資料集 image_id 可能需要補零，先跳過不存在的
            continue

        sample = {
            "id": count + 1,
            "image_path": image_path.replace("\\", "/"),
            "question": question,
            "answer": gold_answer
        }

        samples.append(sample)
        count += 1

        if count >= MAX_SAMPLES:
            break

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(samples)} samples to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()