#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import io
import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor

from PIL import Image
from tqdm import tqdm


def norm_text(v):
    return str(v or "").strip().lower()


def norm_seq(v):
    if isinstance(v, list):
        return tuple(v)
    return v


def dedup_affordances(affs):
    out = []
    seen = set()
    for aff in affs or []:
        if not isinstance(aff, dict):
            continue
        key = (norm_text(aff.get("action")), norm_seq(aff.get("point")))
        if key in seen:
            continue
        seen.add(key)
        out.append(aff)
    return out


def dedup_parts(parts):
    out = []
    seen = set()
    for p in parts or []:
        if not isinstance(p, dict):
            continue
        p = dict(p)
        p["affordances"] = dedup_affordances(p.get("affordances", []))
        key = (
            norm_text(p.get("part_name")),
            norm_seq(p.get("bbox") or p.get("part_bbox")),
            json.dumps(p.get("affordances", []), ensure_ascii=False, sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def dedup_objects(objs):
    out = []
    seen = set()
    for obj in objs or []:
        if not isinstance(obj, dict):
            continue
        obj = dict(obj)
        obj["parts"] = dedup_parts(obj.get("parts", []))
        key = (
            norm_text(obj.get("name") or obj.get("phrase")),
            norm_seq(obj.get("bbox") or obj.get("object_bbox")),
            json.dumps(obj.get("parts", []), ensure_ascii=False, sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(obj)
    return out


def build_affordance_entry(aff):
    action = aff.get("action")
    sampled_points = aff.get("sampled_points") or []
    point = sampled_points[0] if sampled_points else None
    return {
        "action": action,
        "point": point,
    }


def build_part_entry(part):
    def pick_point(aff):
        # Strict rule for obj-part-aff placeholder data:
        # always read affordance point from sampled_points[0].
        sps = aff.get("sampled_points") or []
        if sps and isinstance(sps[0], (list, tuple)) and len(sps[0]) >= 2:
            return [sps[0][0], sps[0][1]]
        return None

    affordances = []
    # Compatible with strict schema where affordances are nested under parts.
    for aff in (part.get("affordances") or []):
        if not isinstance(aff, dict):
            continue
        point = pick_point(aff)
        affordances.append(
            {
                "action": aff.get("action"),
                "point": point,
            }
        )
    return {
        "part_name": part.get("part_name"),
        "bbox": part.get("part_bbox") or part.get("bbox"),
        "affordances": affordances,
    }


def build_object_entry(obj):
    parts = [build_part_entry(part) for part in (obj.get("parts") or [])]
    out = {
        "name": obj.get("phrase") or obj.get("name"),
        "bbox": obj.get("object_bbox") or obj.get("bbox"),
        "parts": parts,
    }
    return out


def convert_record_to_annotation(data):
    objects = [build_object_entry(obj) for obj in (data.get("objects") or [])]
    objects = dedup_objects(objects)
    return {"objects": objects}


def process_one(line):
    try:
        line = line.strip()
        if not line:
            return None

        data = json.loads(line)
        image_path = data.get("image_path")
        if not image_path or not os.path.exists(image_path):
            return None

        annotation = convert_record_to_annotation(data)

        image = Image.open(image_path).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        ann_json = json.dumps(annotation, ensure_ascii=False)
        return img_base64, ann_json
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to TSV with multiprocessing."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="",
    )
    parser.add_argument("--save_image_tsv_path", type=str, required=True)
    parser.add_argument("--save_ann_tsv_path", type=str, required=True)
    parser.add_argument("--save_ann_lineidx_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--max_inflight", type=int, default=4096)
    parser.add_argument("--chunksize", type=int, default=64)
    args = parser.parse_args()

    for path in [
        args.save_image_tsv_path,
        args.save_ann_tsv_path,
        args.save_ann_lineidx_path,
    ]:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    f_image = open(args.save_image_tsv_path, "wb")
    f_ann = open(args.save_ann_tsv_path, "wb")
    f_idx = open(args.save_ann_lineidx_path, "wb")

    img_offset = 0
    ann_offset = 0
    processed = 0
    written = 0
    futures = deque()

    def submit_batch(executor, batch):
        for line in batch:
            futures.append(executor.submit(process_one, line))

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        batch = []
        with open(args.json_file, "r", encoding="utf-8") as json_f:
            pbar = tqdm(desc="Converting SceneParser JSONL to TSV (dedup)")
            for line in json_f:
                processed += 1
                batch.append(line)

                if len(batch) >= args.chunksize:
                    submit_batch(executor, batch)
                    batch = []

                while len(futures) >= args.max_inflight:
                    result = futures.popleft().result()
                    pbar.update(1)
                    if result is None:
                        continue

                    img_base64, ann_json = result
                    img_line = f"{img_offset}\t{img_base64}\n".encode("utf-8")
                    ann_line = f"{img_offset}\t{ann_json}\n".encode("utf-8")
                    idx_line = f"{ann_offset}\n".encode("utf-8")

                    f_image.write(img_line)
                    f_ann.write(ann_line)
                    f_idx.write(idx_line)

                    img_offset += len(img_line)
                    ann_offset += len(ann_line)
                    written += 1

            if batch:
                submit_batch(executor, batch)

            while futures:
                result = futures.popleft().result()
                pbar.update(1)
                if result is None:
                    continue

                img_base64, ann_json = result
                img_line = f"{img_offset}\t{img_base64}\n".encode("utf-8")
                ann_line = f"{img_offset}\t{ann_json}\n".encode("utf-8")
                idx_line = f"{ann_offset}\n".encode("utf-8")

                f_image.write(img_line)
                f_ann.write(ann_line)
                f_idx.write(idx_line)

                img_offset += len(img_line)
                ann_offset += len(ann_line)
                written += 1

            pbar.close()

    f_image.close()
    f_ann.close()
    f_idx.close()

    print("Conversion finished (dedup)")
    print("Processed:", processed)
    print("Written:", written)
    print("Image TSV:", args.save_image_tsv_path)
    print("Annotation TSV:", args.save_ann_tsv_path)
    print("Lineidx:", args.save_ann_lineidx_path)


if __name__ == "__main__":
    main()
