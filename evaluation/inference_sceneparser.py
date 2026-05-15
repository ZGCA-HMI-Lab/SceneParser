import argparse
import json
import os
import re

from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

DEFAULT_PROMPT = "You are a helpful assistant"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_jsonl_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument(
        "--question",
        type=str,
        default="Parse [OBJ] in this image and return its hierarchical structure with object bbox, part bboxes, and affordance points.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=2560 * 28 * 28)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--cross_image_batch_size",
        type=int,
        default=1,
        help="How many prompt items to accumulate across images before a batched generate call. 1 preserves the old behavior.",
    )
    return parser.parse_args()


def build_llm_input(image, prompt, system_prompt, min_pixels, max_pixels):
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    image_inputs, _ = process_vision_info(messages)
    return {
        "prompt": processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ),
        "multi_modal_data": {"image": image_inputs},
    }


def batched_inference(image, prompts, system_prompt, min_pixels, max_pixels):
    llm_inputs = [
        build_llm_input(image, prompt, system_prompt, min_pixels, max_pixels)
        for prompt in prompts
    ]
    outputs = model.generate(llm_inputs, sampling_params=sampling_params)
    return [out.outputs[0].text for out in outputs]


def batched_inference_from_inputs(llm_inputs):
    outputs = model.generate(llm_inputs, sampling_params=sampling_params)
    return [out.outputs[0].text for out in outputs]


def extract_json_string(text):
    text = text.split("<|im_end|>")[0]
    code_block = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if code_block:
        return code_block[0]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return ""


def token_box_to_abs(box_str, w, h):
    # 1) RexOmni token format: "<x0><y0><x1><y1>"
    if isinstance(box_str, str):
        nums = re.findall(r"<(\d+)>", box_str)
        if len(nums) == 4:
            x0, y0, x1, y1 = [int(n) for n in nums]
            return [x0 / 999.0 * w, y0 / 999.0 * h, x1 / 999.0 * w, y1 / 999.0 * h]
        return None

    # 2) Robust compatibility: [x0, y0, x1, y1]
    if isinstance(box_str, (list, tuple)) and len(box_str) == 4:
        try:
            vals = [float(v) for v in box_str]
        except (TypeError, ValueError):
            return None

        m = max(vals) if vals else 0.0
        if m <= 1.0:
            # normalized coords
            x0, y0, x1, y1 = vals
            return [x0 * w, y0 * h, x1 * w, y1 * h]
        if m <= 999.0:
            # token-like numeric coords
            x0, y0, x1, y1 = vals
            return [x0 / 999.0 * w, y0 / 999.0 * h, x1 / 999.0 * w, y1 / 999.0 * h]
        # already absolute pixel coords
        return vals

    return None


def token_point_to_abs(point_str, w, h):
    # 1) RexOmni token format: "<x><y>"
    if isinstance(point_str, str):
        nums = re.findall(r"<(\d+)>", point_str)
        if len(nums) == 2:
            x, y = [int(n) for n in nums]
            return [x / 999.0 * w, y / 999.0 * h]
        return None

    # 2) Robust compatibility: [x, y]
    if isinstance(point_str, (list, tuple)) and len(point_str) == 2:
        try:
            x, y = float(point_str[0]), float(point_str[1])
        except (TypeError, ValueError):
            return None
        m = max(abs(x), abs(y))
        if m <= 1.0:
            # normalized coords
            return [x * w, y * h]
        if m <= 999.0:
            # token-like numeric coords
            return [x / 999.0 * w, y / 999.0 * h]
        # already absolute pixel coords
        return [x, y]

    return None


def parse_structured_prediction(text, w, h):
    json_str = extract_json_string(text)
    if not json_str:
        return {"objects": []}
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {"objects": []}

    raw_objects = data.get("objects", [])
    if not isinstance(raw_objects, list):
        return {"objects": []}

    objects = []
    for obj in raw_objects:
        if not isinstance(obj, dict):
            continue

        pred_obj = {
            "name": obj.get("name"),
            "bbox": token_box_to_abs(obj.get("bbox"), w, h),
            "has_part": obj.get("has_part"),
            "has_affordance": obj.get("has_affordance"),
            "parts": [],
            "affordances": [],
        }

        raw_parts = obj.get("parts", [])
        if isinstance(raw_parts, list):
            for part in raw_parts:
                if not isinstance(part, dict):
                    continue
                pred_part = {
                    "part_name": part.get("part_name"),
                    "bbox": token_box_to_abs(part.get("bbox"), w, h),
                    "affordances": [],
                }
                raw_part_affs = part.get("affordances", [])
                if isinstance(raw_part_affs, list):
                    for aff in raw_part_affs:
                        if not isinstance(aff, dict):
                            continue
                        pred_aff = {
                            "action": aff.get("action"),
                            "point": token_point_to_abs(aff.get("point"), w, h),
                        }
                        pred_part["affordances"].append(pred_aff)
                pred_obj["parts"].append(pred_part)
        # Strict schema: only object->parts->affordances is supported.
        # object-level affordances are intentionally ignored.

        objects.append(pred_obj)
    return {"objects": objects}


def build_gt_for_category(record, category):
    gt_objects = []
    for obj in record.get("objects", []):
        if obj.get("phrase") != category:
            continue
        gt_parts = []
        for part in obj.get("parts", []) or []:
            if part.get("part_name") and part.get("part_bbox") is not None:
                part_affordances = []
                for aff in part.get("affordances", []) or []:
                    sampled_points = aff.get("sampled_points") or []
                    if aff.get("action") and (aff.get("affordance_bbox") is not None or sampled_points):
                        part_affordances.append(
                            {
                                "action": aff.get("action"),
                                "affordance_bbox": aff.get("affordance_bbox"),
                                "mask_path": aff.get("mask_path"),
                                "points": sampled_points,
                                "point": sampled_points[0] if sampled_points else None,
                            }
                        )
                gt_parts.append(
                    {
                        "part_name": part.get("part_name"),
                        "bbox": part.get("part_bbox"),
                        "affordances": part_affordances,
                    }
                )
        has_part_affordance = any((part.get("affordances") or []) for part in gt_parts)
        gt_objects.append(
            {
                "name": obj.get("phrase"),
                "bbox": obj.get("object_bbox"),
                "has_part": len(gt_parts) > 0,
                "has_affordance": has_part_affordance,
                "parts": gt_parts,
                # Strict schema: affordances are evaluated only under parts.
                "affordances": [],
            }
        )
    return {"objects": gt_objects}


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    model = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tokenizer_mode="slow",
        limit_mm_per_prompt={"image": 10, "video": 10},
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=0.8,
        repetition_penalty=1.05,
        top_k=1,
        temperature=0.0,
        skip_special_tokens=False,
        stop=["<|im_end|>", "<0><0><0><0>"],
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )

    records = load_records(args.test_jsonl_path)
    if args.end_idx == -1:
        records = records[args.start_idx :]
    else:
        records = records[args.start_idx : args.end_idx]

    predictions = []
    pending_samples = []

    def flush_pending():
        global predictions
        nonlocal_pending = pending_samples[:]
        if not nonlocal_pending:
            return []
        llm_inputs = [sample["llm_input"] for sample in nonlocal_pending]
        outputs = batched_inference_from_inputs(llm_inputs)
        flushed_predictions = []
        for sample, output in zip(nonlocal_pending, outputs):
            flushed_predictions.append(
                {
                    "image_path": sample["image_path"],
                    "category": sample["category"],
                    "question": sample["question"],
                    "gt": sample["gt"],
                    "extracted_predictions": parse_structured_prediction(
                        output, sample["width"], sample["height"]
                    ),
                    "raw_response": output,
                    "task_name": "sceneparser_structured_parsing",
                    "dataset_name": "SceneParserStructured",
                }
            )
        return flushed_predictions

    for record in tqdm(records, desc="Structured SceneParser inference"):
        image_path = record["image_path"]
        if not os.path.exists(image_path):
            continue
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        categories = sorted(
            {
                obj.get("phrase")
                for obj in record.get("objects", [])
                if obj.get("phrase")
            }
        )
        if not categories:
            continue

        questions = [args.question.replace("[OBJ]", category) for category in categories]

        for category, question in zip(categories, questions):
            pending_samples.append(
                {
                    "image_path": image_path,
                    "category": category,
                    "question": question,
                    "gt": build_gt_for_category(record, category),
                    "width": w,
                    "height": h,
                    "llm_input": build_llm_input(
                        image,
                        question,
                        args.system_prompt,
                        args.min_pixels,
                        args.max_pixels,
                    ),
                }
            )

        if len(pending_samples) >= args.cross_image_batch_size:
            predictions.extend(flush_pending())
            pending_samples = []

    if pending_samples:
        predictions.extend(flush_pending())

    with open(args.save_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
