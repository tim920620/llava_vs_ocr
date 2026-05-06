import os
import re
import json
import csv
import argparse
from collections import Counter
from typing import List, Dict, Any
from difflib import SequenceMatcher

import torch
import easyocr
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s$%.\-]", "", text)
    return text


def exact_match_score(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def accuracy_score_simple(pred: str, gold: str) -> int:
    return exact_match_score(pred, gold)


def token_f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def anls_score(pred: str, gold: str, threshold: float = 0.8) -> float:
    """
    ANLS = Average Normalized Levenshtein Similarity 的簡化近似版
    這裡用 SequenceMatcher 作為字串相似度近似。
    若相似度低於 threshold，直接視為 0。
    """
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)

    if pred_norm == "" and gold_norm == "":
        return 1.0
    if pred_norm == "" or gold_norm == "":
        return 0.0

    sim = SequenceMatcher(None, pred_norm, gold_norm).ratio()

    if sim < threshold:
        return 0.0
    return sim


def weighted_score(pred: str, gold: str) -> float:
    """
    綜合加權分數：
    - 全對給高分
    - 部分正確按比例給分
    - 不使用懲罰機制
    """
    em = exact_match_score(pred, gold)
    f1 = token_f1_score(pred, gold)
    anls = anls_score(pred, gold)

    # score = 0.8 * em + 0.1 * f1 + 0.1 * anls
    # score = max(anls, f1) + em

    # 限制分數範圍在 0~1
    # score = max(0.0, min(1.0, score))
    # score = max(f1, anls)

    # if em == 1:
    #     score = min(1.0, score + 0.1)
    score = anls * 1.5 + em * 0.5
    return score


class OCRExtractor:
    def __init__(self, lang_list: List[str] = None, use_gpu: bool = True):
        if lang_list is None:
            lang_list = ["en"]

        self.reader = easyocr.Reader(
            lang_list,
            gpu=use_gpu
        )

    def extract_text(self, image_path: str) -> str:
        try:
            result = self.reader.readtext(image_path, detail=0)
            if not result:
                return ""
            texts = [text.strip() for text in result if isinstance(text, str) and text.strip()]
            return " ".join(texts)
        except Exception as e:
            print(f"[WARNING] OCR failed for {image_path}: {e}")
            return ""


class LlavaVQA:
    def __init__(self, model_id: str, max_new_tokens: int = 64):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

        print(f"[INFO] device = {self.device}")
        print(f"[INFO] torch_dtype = {self.torch_dtype}")
        print(f"[INFO] loading model: {self.model_id}")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.model.eval()

    def ask(self, image_path: str, prompt_text: str) -> str:
        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )

        input_token_len = inputs["input_ids"].shape[1]
        generated_ids = output[:, input_token_len:]

        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        return answer.strip()


def build_baseline_prompt(question: str) -> str:
    return (
        "Answer the question based on the image.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_ocr_prompt(question: str, ocr_text: str) -> str:
    if not ocr_text.strip():
        ocr_text = "[NO OCR TEXT FOUND]"

    #return (
    #    "You are given an image and OCR results extracted from the image.\n\n"
    #    f"OCR results:\n{ocr_text}\n\n"
    #    "Answer the question based on the image and OCR results briefly.\n"
    #    f"Question: {question}\n"
    #    "Answer:"
    #)

    return (
    "You are given an image and OCR results extracted from the image.\n\n"
    f"OCR results:\n{ocr_text}\n\n"
    "Answer the question primarily based on the image. "
    "Use OCR results only as supporting information when needed. "
    "If there is any conflict, prioritize the image. "
    "Respond briefly with the final answer only.\n"
    f"Question: {question}\n"
    "Answer:"
    )


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of samples.")

    required_keys = {"id", "image_path", "question", "answer"}
    for i, sample in enumerate(data):
        missing = required_keys - set(sample.keys())
        if missing:
            raise ValueError(f"Sample index {i} is missing keys: {missing}")

    return data


def save_results(results: List[Dict[str, Any]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "results.json")
    csv_path = os.path.join(output_dir, "results.csv")
    summary_path = os.path.join(output_dir, "summary.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if results:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    def avg(key: str) -> float:
        if not results:
            return 0.0
        return sum(r[key] for r in results) / len(results)

    summary = {
        "num_samples": len(results),
        "baseline": {
            "accuracy": avg("baseline_acc"),
            "exact_match": avg("baseline_em"),
            "f1": avg("baseline_f1"),
            "anls": avg("baseline_anls"),
            "weighted_score": avg("baseline_weighted")
        },
        "ocr_assisted": {
            "accuracy": avg("ocr_acc"),
            "exact_match": avg("ocr_em"),
            "f1": avg("ocr_f1"),
            "anls": avg("ocr_anls"),
            "weighted_score": avg("ocr_weighted")
        }
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[INFO] Results saved to: {output_dir}")


def run_experiment(
    dataset_path: str,
    output_dir: str,
    model_id: str,
    ocr_lang: str,
    max_new_tokens: int
):
    samples = load_dataset(dataset_path)
    use_gpu_for_ocr = torch.cuda.is_available()

    print(f"[INFO] OCR use_gpu = {use_gpu_for_ocr}")

    ocr_extractor = OCRExtractor(lang_list=[ocr_lang], use_gpu=use_gpu_for_ocr)
    llava_model = LlavaVQA(model_id=model_id, max_new_tokens=max_new_tokens)

    results = []

    for sample in tqdm(samples, desc="Running Experiment"):
        sample_id = sample["id"]
        image_path = sample["image_path"]
        question = sample["question"]
        gold_answer = sample["answer"]

        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found: {image_path}")
            continue

        try:
            baseline_prompt = build_baseline_prompt(question)
            baseline_pred = llava_model.ask(image_path, baseline_prompt)

            ocr_text = ocr_extractor.extract_text(image_path)
            ocr_prompt = build_ocr_prompt(question, ocr_text)
            ocr_pred = llava_model.ask(image_path, ocr_prompt)

            baseline_em = exact_match_score(baseline_pred, gold_answer)
            baseline_f1 = token_f1_score(baseline_pred, gold_answer)
            baseline_anls = anls_score(baseline_pred, gold_answer)
            baseline_weighted = weighted_score(baseline_pred, gold_answer)

            ocr_em = exact_match_score(ocr_pred, gold_answer)
            ocr_f1 = token_f1_score(ocr_pred, gold_answer)
            ocr_anls = anls_score(ocr_pred, gold_answer)
            ocr_weighted = weighted_score(ocr_pred, gold_answer)

            row = {
                "id": sample_id,
                "image_path": image_path,
                "question": question,
                "gold_answer": gold_answer,

                "baseline_pred": baseline_pred,
                "baseline_acc": accuracy_score_simple(baseline_pred, gold_answer),
                "baseline_em": baseline_em,
                "baseline_f1": baseline_f1,
                "baseline_anls": baseline_anls,
                "baseline_weighted": baseline_weighted,

                "ocr_text": ocr_text,
                "ocr_pred": ocr_pred,
                "ocr_acc": accuracy_score_simple(ocr_pred, gold_answer),
                "ocr_em": ocr_em,
                "ocr_f1": ocr_f1,
                "ocr_anls": ocr_anls,
                "ocr_weighted": ocr_weighted,
            }

            results.append(row)

        except Exception as e:
            print(f"[ERROR] Failed on sample id={sample_id}, image={image_path}")
            print(f"        {e}")

    save_results(results, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/samples.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--ocr_lang", type=str, default="en")
    parser.add_argument("--max_new_tokens", type=int, default=64)

    args = parser.parse_args()

    run_experiment(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_id=args.model_id,
        ocr_lang=args.ocr_lang,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()