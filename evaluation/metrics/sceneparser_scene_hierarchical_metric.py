#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image


CANONICAL_EVAL = False
PART_ALIAS_MAP = {}
ACTION_ALIAS_MAP = {}
MASK_CACHE = {}


def normalize_label(value):
    return str(value or "").strip().lower()


def load_alias_map(mapping_path):
    if not mapping_path:
        return {}
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {mapping_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {normalize_label(k): normalize_label(v) for k, v in data.get("alias_map", {}).items()}


def canonicalize_part_name(value):
    name = normalize_label(value)
    if not CANONICAL_EVAL:
        return name
    return PART_ALIAS_MAP.get(name, name)


def canonicalize_action(value):
    action = normalize_label(value)
    if not CANONICAL_EVAL:
        return action
    return ACTION_ALIAS_MAP.get(action, action)


def canonicalize_object_name(value):
    return normalize_label(value)


def iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0


def is_point_in_bbox(point, bbox):
    if point is None or bbox is None:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return x1 <= x <= x2 and y1 <= y <= y2


def point_in_mask(point, mask_path):
    if not mask_path:
        return None
    mask = MASK_CACHE.get(mask_path)
    if mask is None and mask_path not in MASK_CACHE:
        try:
            arr = np.array(Image.open(mask_path))
            if arr.ndim == 3:
                arr = arr.max(axis=2)
            mask = arr > 0
            MASK_CACHE[mask_path] = mask
        except Exception:
            MASK_CACHE[mask_path] = None
            return None
    if mask is None:
        return None
    x, y = int(round(point[0])), int(round(point[1]))
    h, w = mask.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return bool(mask[y, x])


def get_aff_bbox(aff, fallback_bbox=None):
    bbox = aff.get("affordance_bbox", aff.get("bbox"))
    if bbox is None:
        bbox = fallback_bbox
    return bbox


def is_point_in_aff_region(point, gt_aff, fallback_bbox=None):
    mask_path = gt_aff.get("mask_path")
    if mask_path:
        in_mask = point_in_mask(point, mask_path)
        if in_mask is not None:
            return in_mask
    return is_point_in_bbox(point, get_aff_bbox(gt_aff, fallback_bbox=fallback_bbox))


def get_part_affordances(part):
    affs = []
    for aff in part.get("affordances", []) or []:
        if isinstance(aff, dict) and aff.get("action"):
            affs.append(aff)
    return affs


def get_object_affordances(obj):
    affs = []
    for part in obj.get("parts", []) or []:
        part_bbox = part.get("bbox")
        for aff in get_part_affordances(part):
            item = dict(aff)
            item.setdefault("part_bbox", part_bbox)
            affs.append(item)
    for aff in obj.get("affordances", []) or []:
        if isinstance(aff, dict) and aff.get("action"):
            affs.append(aff)
    return affs


def object_has_parts(obj):
    return bool(obj.get("parts", []) or [])


def object_has_part_affordances(obj):
    return any(bool(get_part_affordances(part)) for part in (obj.get("parts", []) or []))


def object_has_any_affordances(obj):
    return bool(get_object_affordances(obj))


def matched_object_loose_parse_success(gt_obj, pred_obj):
    gt_has_parts = object_has_parts(gt_obj)
    gt_has_part_affs = object_has_part_affordances(gt_obj)
    gt_has_any_affs = object_has_any_affordances(gt_obj)

    pred_has_parts = object_has_parts(pred_obj)
    pred_has_part_affs = object_has_part_affordances(pred_obj)
    pred_has_any_affs = object_has_any_affordances(pred_obj)

    if gt_has_parts and gt_has_part_affs:
        return pred_has_parts and pred_has_part_affs
    if gt_has_parts:
        return pred_has_parts
    if gt_has_any_affs:
        return pred_has_any_affs
    return False


def matched_object_strict_parse_success(gt_obj, pred_obj):
    gt_has_parts = object_has_parts(gt_obj)
    gt_has_part_affs = object_has_part_affordances(gt_obj)
    gt_has_any_affs = object_has_any_affordances(gt_obj)

    pred_has_parts = object_has_parts(pred_obj)
    pred_has_part_affs = object_has_part_affordances(pred_obj)
    pred_has_any_affs = object_has_any_affordances(pred_obj)

    if gt_has_parts and gt_has_part_affs:
        return pred_has_parts and pred_has_part_affs
    if gt_has_parts:
        return pred_has_parts and (not pred_has_any_affs)
    if gt_has_any_affs:
        return pred_has_any_affs and (not pred_has_parts)
    return False


def get_aff_point(aff):
    if aff is None:
        return None
    p = aff.get("point")
    if isinstance(p, list) and len(p) >= 2:
        return [p[0], p[1]]
    points = aff.get("points") or aff.get("sampled_points") or []
    if points and isinstance(points[0], list) and len(points[0]) >= 2:
        return [points[0][0], points[0][1]]
    return None


def summarize_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def harmonic3(a, b, c, eps=1e-12):
    return 3.0 / ((1.0 / (a + eps)) + (1.0 / (b + eps)) + (1.0 / (c + eps)))


def match_objects_scene(gt_objects, pred_objects, thr):
    matches = []
    used_pred = set()
    for gi, gt in enumerate(gt_objects):
        gt_name = canonicalize_object_name(gt.get("name"))
        best_j = -1
        best_iou = 0.0
        for pj, pred in enumerate(pred_objects):
            if pj in used_pred:
                continue
            if canonicalize_object_name(pred.get("name")) != gt_name:
                continue
            score = iou(gt.get("bbox"), pred.get("bbox"))
            if score >= thr and score > best_iou:
                best_iou = score
                best_j = pj
        if best_j != -1:
            used_pred.add(best_j)
            matches.append((gi, best_j))
    return matches


def match_parts_in_objects(gt_parts, pred_parts, thr):
    used_pred = set()
    pairs = []
    for gi, gt in enumerate(gt_parts):
        gt_name = canonicalize_part_name(gt.get("part_name"))
        best_j = -1
        best_iou = 0.0
        for pj, pred in enumerate(pred_parts):
            if pj in used_pred:
                continue
            if canonicalize_part_name(pred.get("part_name")) != gt_name:
                continue
            score = iou(gt.get("bbox"), pred.get("bbox"))
            if score >= thr and score > best_iou:
                best_iou = score
                best_j = pj
        if best_j != -1:
            used_pred.add(best_j)
            pairs.append((gi, best_j))
    return pairs


def match_affordances_in_part_pairs(gt_parts, pred_parts, part_pairs):
    gt_total = 0
    pred_total = 0
    matched = 0
    used_pred = set()

    for part in gt_parts:
        gt_total += len(get_part_affordances(part))
    for part in pred_parts:
        pred_total += len(get_part_affordances(part))

    for g_idx, p_idx in part_pairs:
        gt_affs = get_part_affordances(gt_parts[g_idx])
        pred_affs = get_part_affordances(pred_parts[p_idx])
        fallback_bbox = gt_parts[g_idx].get("bbox")
        for gt_aff in gt_affs:
            gt_action = canonicalize_action(gt_aff.get("action"))
            best_j = -1
            for j, pred_aff in enumerate(pred_affs):
                uid = (p_idx, j)
                if uid in used_pred:
                    continue
                if canonicalize_action(pred_aff.get("action")) != gt_action:
                    continue
                if is_point_in_aff_region(get_aff_point(pred_aff), gt_aff, fallback_bbox=fallback_bbox):
                    best_j = j
                    break
            if best_j != -1:
                used_pred.add((p_idx, best_j))
                matched += 1
    return matched, gt_total, pred_total


def safe_mean(values):
    return mean(values) if values else 0.0


def summarize(scores):
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = mean([s[0] for s in scores])
    r = mean([s[1] for s in scores])
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def evaluate(samples):
    # non-e2e (macro/conditional-style, aligned with obj-level naming)
    obj_scores_05 = []
    obj_scores_095 = []
    part_scores_05 = []
    part_scores_095 = []
    aff_scores_05 = []
    aff_scores_095 = []

    # e2e (global count-based)
    object_counts = {0.5: {"tp": 0, "fp": 0, "fn": 0}, 0.95: {"tp": 0, "fp": 0, "fn": 0}}
    part_counts = {0.5: {"tp": 0, "fp": 0, "fn": 0}, 0.95: {"tp": 0, "fp": 0, "fn": 0}}
    aff_counts = {0.5: {"tp": 0, "fp": 0, "fn": 0}, 0.95: {"tp": 0, "fp": 0, "fn": 0}}

    scene_object_recall_05 = []
    scene_part_recall_05 = []
    scene_aff_recall_05 = []
    scene_exact_success_05 = []
    parse_eligible_gt_objects_05 = 0
    scene_loose_parse_success_count_05 = 0
    scene_strict_parse_success_count_05 = 0
    scene_hier_em_success_count_05 = 0

    for sample in samples:
        gt_objects = sample.get("gt", {}).get("objects", []) or []
        pred_objects = sample.get("extracted_predictions", {}).get("objects", []) or []

        object_matches_05 = None
        part_recall_img = None
        aff_recall_img = None

        for thr in (0.5, 0.95):
            obj_matches = match_objects_scene(gt_objects, pred_objects, thr)
            tp_obj = len(obj_matches)
            fp_obj = max(0, len(pred_objects) - tp_obj)
            fn_obj = max(0, len(gt_objects) - tp_obj)

            # non-e2e object scores (sample-level P/R)
            obj_p = tp_obj / len(pred_objects) if len(pred_objects) else (1.0 if len(gt_objects) == 0 else 0.0)
            obj_r = tp_obj / len(gt_objects) if len(gt_objects) else 1.0
            if thr == 0.5:
                obj_scores_05.append((obj_p, obj_r))
            else:
                obj_scores_095.append((obj_p, obj_r))

            # e2e object counts
            object_counts[thr]["tp"] += tp_obj
            object_counts[thr]["fp"] += fp_obj
            object_counts[thr]["fn"] += fn_obj

            # e2e totals use all objects in the scene
            gt_part_total_all = sum(len(obj.get("parts", []) or []) for obj in gt_objects)
            pred_part_total_all = sum(len(obj.get("parts", []) or []) for obj in pred_objects)
            gt_aff_total_all = sum(
                len(get_part_affordances(part))
                for obj in gt_objects
                for part in (obj.get("parts", []) or [])
            )
            pred_aff_total_all = sum(
                len(get_part_affordances(part))
                for obj in pred_objects
                for part in (obj.get("parts", []) or [])
            )

            part_tp = 0
            aff_tp = 0

            # non-e2e part/aff sample-level collections (conditional on matched objects)
            part_prs = []
            aff_prs = []

            for gi, pj in obj_matches:
                gt_obj = gt_objects[gi]
                pred_obj = pred_objects[pj]
                gt_parts = gt_obj.get("parts", []) or []
                pred_parts = pred_obj.get("parts", []) or []

                part_pairs = match_parts_in_objects(gt_parts, pred_parts, thr)
                part_tp += len(part_pairs)

                gt_part_n = len(gt_parts)
                pred_part_n = len(pred_parts)
                if not (gt_part_n == 0 and pred_part_n == 0):
                    p_part = len(part_pairs) / pred_part_n if pred_part_n else 0.0
                    r_part = len(part_pairs) / gt_part_n if gt_part_n else 0.0
                    part_prs.append((p_part, r_part))

                matched_aff, gt_aff_n, pred_aff_n = match_affordances_in_part_pairs(
                    gt_parts, pred_parts, part_pairs
                )
                aff_tp += matched_aff
                if not (gt_aff_n == 0 and pred_aff_n == 0):
                    p_aff = matched_aff / pred_aff_n if pred_aff_n else 0.0
                    r_aff = matched_aff / gt_aff_n if gt_aff_n else 0.0
                    aff_prs.append((p_aff, r_aff))

            # non-e2e part/aff summaries aggregated by sample
            if part_prs:
                part_p = mean([x[0] for x in part_prs])
                part_r = mean([x[1] for x in part_prs])
                if thr == 0.5:
                    part_scores_05.append((part_p, part_r))
                else:
                    part_scores_095.append((part_p, part_r))
            if aff_prs:
                aff_p = mean([x[0] for x in aff_prs])
                aff_r = mean([x[1] for x in aff_prs])
                if thr == 0.5:
                    aff_scores_05.append((aff_p, aff_r))
                else:
                    aff_scores_095.append((aff_p, aff_r))

            # e2e part/aff counts
            part_fp = max(0, pred_part_total_all - part_tp)
            part_fn = max(0, gt_part_total_all - part_tp)
            aff_fp = max(0, pred_aff_total_all - aff_tp)
            aff_fn = max(0, gt_aff_total_all - aff_tp)

            part_counts[thr]["tp"] += part_tp
            part_counts[thr]["fp"] += part_fp
            part_counts[thr]["fn"] += part_fn
            aff_counts[thr]["tp"] += aff_tp
            aff_counts[thr]["fp"] += aff_fp
            aff_counts[thr]["fn"] += aff_fn

            if thr == 0.5:
                object_matches_05 = obj_matches
                part_recall_img = (part_tp / gt_part_total_all) if gt_part_total_all else 1.0
                aff_recall_img = (aff_tp / gt_aff_total_all) if gt_aff_total_all else 1.0

        obj_recall_img = (len(object_matches_05) / len(gt_objects)) if gt_objects else 1.0
        scene_object_recall_05.append(obj_recall_img)
        scene_part_recall_05.append(part_recall_img if part_recall_img is not None else 1.0)
        scene_aff_recall_05.append(aff_recall_img if aff_recall_img is not None else 1.0)
        scene_exact_success_05.append(
            1.0 if (obj_recall_img == 1.0 and part_recall_img == 1.0 and aff_recall_img == 1.0) else 0.0
        )

        matched_pred_by_gt = {gi: pj for gi, pj in (object_matches_05 or [])}
        for gi, gt_obj in enumerate(gt_objects):
            if not (object_has_parts(gt_obj) or object_has_any_affordances(gt_obj)):
                continue
            parse_eligible_gt_objects_05 += 1
            pj = matched_pred_by_gt.get(gi)
            if pj is None:
                continue
            pred_obj = pred_objects[pj]
            scene_loose_parse_success_count_05 += int(
                matched_object_loose_parse_success(gt_obj, pred_obj)
            )
            scene_strict_parse_success_count_05 += int(
                matched_object_strict_parse_success(gt_obj, pred_obj)
            )

            gt_parts = gt_obj.get("parts", []) or []
            pred_parts = pred_obj.get("parts", []) or []
            part_pairs = match_parts_in_objects(gt_parts, pred_parts, 0.5)
            matched_aff, gt_aff_n, pred_aff_n = match_affordances_in_part_pairs(
                gt_parts, pred_parts, part_pairs
            )
            part_tp = len(part_pairs)
            gt_part_n = len(gt_parts)
            pred_part_n = len(pred_parts)
            gt_has_parts = object_has_parts(gt_obj)
            gt_has_part_affs = object_has_part_affordances(gt_obj)
            gt_has_any_affs = object_has_any_affordances(gt_obj)

            strict_success = False
            if gt_has_parts and gt_has_part_affs:
                part_r = (part_tp / gt_part_n) if gt_part_n else 0.0
                part_p = (part_tp / pred_part_n) if pred_part_n else 0.0
                aff_r = (matched_aff / gt_aff_n) if gt_aff_n else 0.0
                aff_p = (matched_aff / pred_aff_n) if pred_aff_n else 0.0
                strict_success = (
                    part_r == 1.0 and part_p == 1.0 and aff_r == 1.0 and aff_p == 1.0
                )
            elif gt_has_parts:
                part_r = (part_tp / gt_part_n) if gt_part_n else 0.0
                part_p = (part_tp / pred_part_n) if pred_part_n else 0.0
                strict_success = (part_r == 1.0 and part_p == 1.0)
            elif gt_has_any_affs:
                gt_affs = get_object_affordances(gt_obj)
                pred_affs = get_object_affordances(pred_obj)
                root_aff_matches = 0
                used_pred = set()
                for gt_aff in gt_affs:
                    best_j = -1
                    for j, pred_aff in enumerate(pred_affs):
                        if j in used_pred:
                            continue
                        if canonicalize_action(pred_aff.get("action")) != canonicalize_action(gt_aff.get("action")):
                            continue
                        if is_point_in_aff_region(
                            get_aff_point(pred_aff),
                            gt_aff,
                            fallback_bbox=gt_aff.get("part_bbox"),
                        ):
                            best_j = j
                            break
                    if best_j != -1:
                        used_pred.add(best_j)
                        root_aff_matches += 1
                strict_success = (
                    root_aff_matches == len(gt_affs)
                    and root_aff_matches == len(pred_affs)
                )
            scene_hier_em_success_count_05 += int(strict_success)

    object_05 = summarize_counts(**object_counts[0.5])
    object_095 = summarize_counts(**object_counts[0.95])
    part_05 = summarize_counts(**part_counts[0.5])
    part_095 = summarize_counts(**part_counts[0.95])
    aff_05 = summarize_counts(**aff_counts[0.5])
    aff_095 = summarize_counts(**aff_counts[0.95])

    obj_summary_05 = summarize(obj_scores_05)
    obj_summary_095 = summarize(obj_scores_095)
    part_summary_05 = summarize(part_scores_05)
    part_summary_095 = summarize(part_scores_095)
    aff_summary_05 = summarize(aff_scores_05)
    aff_summary_095 = summarize(aff_scores_095)

    f_hier_05 = harmonic3(object_05["f1"], part_05["f1"], aff_05["f1"])

    return {
        # non-e2e (obj-level-compatible naming)
        "object_0.5": obj_summary_05,
        "object_0.95": obj_summary_095,
        "part_0.5": part_summary_05,
        "part_0.95": part_summary_095,
        "affordance_point_in_bbox": aff_summary_05,
        "affordance_point_in_bbox_0.95": aff_summary_095,
        "f_hier": {
            "f_obj": obj_summary_05["f1"],
            "f_part_obj": part_summary_05["f1"],
            "f_aff_part": aff_summary_05["f1"],
            "f_hier_h": harmonic3(
                obj_summary_05["f1"], part_summary_05["f1"], aff_summary_05["f1"]
            ),
            "f_hier": harmonic3(
                obj_summary_05["f1"], part_summary_05["f1"], aff_summary_05["f1"]
            ),
        },
        # e2e
        "L1": object_05,
        "L1_0.95": object_095,
        "L2": part_05,
        "L2_0.95": part_095,
        "L3": aff_05,
        "L3_0.95": aff_095,
        "f_hier_scene_e2e": {
            "f_obj_e2e_0.5": object_05["f1"],
            "f_part_e2e_0.5": part_05["f1"],
            "f_aff_e2e_0.5": aff_05["f1"],
            "f_hier_e2e_0.5": f_hier_05,
        },
        "scene_recall_0.5": {
            "object": object_05["recall"],
            "part": part_05["recall"],
            "affordance": aff_05["recall"],
        },
        "scene_exact_success_rate_0.5": safe_mean(scene_exact_success_05),
        "scene_hier_em_0.5": (
            (scene_hier_em_success_count_05 / parse_eligible_gt_objects_05)
            if parse_eligible_gt_objects_05
            else 0.0
        ),
        "ParseRate": (
            (scene_loose_parse_success_count_05 / parse_eligible_gt_objects_05)
            if parse_eligible_gt_objects_05
            else 0.0
        ),
        "scene_strict_parse_rate_0.5": (
            (scene_strict_parse_success_count_05 / parse_eligible_gt_objects_05)
            if parse_eligible_gt_objects_05
            else 0.0
        ),
        "scene_average_recall_0.5": {
            "object": safe_mean(scene_object_recall_05),
            "part": safe_mean(scene_part_recall_05),
            "affordance": safe_mean(scene_aff_recall_05),
        },
        "diagnostics": {
            "num_samples": len(samples),
            "parse_eligible_gt_objects_0.5": parse_eligible_gt_objects_05,
            "scene_exact_success_count_0.5": int(sum(scene_exact_success_05)),
            "scene_hier_em_success_count_0.5": scene_hier_em_success_count_05,
            "scene_loose_parse_success_count_0.5": scene_loose_parse_success_count_05,
            "scene_strict_parse_success_count_0.5": scene_strict_parse_success_count_05,
            "canonical_eval_enabled": CANONICAL_EVAL,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scene-level SceneParser hierarchical evaluation with full-scene matching."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--canonical_eval", action="store_true")
    parser.add_argument("--part_mapping_path", type=str, default=None)
    parser.add_argument("--action_mapping_path", type=str, default=None)
    return parser.parse_args()


def main():
    global CANONICAL_EVAL, PART_ALIAS_MAP, ACTION_ALIAS_MAP
    args = parse_args()
    CANONICAL_EVAL = args.canonical_eval
    if CANONICAL_EVAL:
        PART_ALIAS_MAP = load_alias_map(args.part_mapping_path) if args.part_mapping_path else {}
        ACTION_ALIAS_MAP = load_alias_map(args.action_mapping_path) if args.action_mapping_path else {}

    with open(args.data_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    results = evaluate(samples)
    Path(args.output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
