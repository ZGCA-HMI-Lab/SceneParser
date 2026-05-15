import argparse
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image


CANONICAL_EVAL = False
PART_ALIAS_MAP = {}
ACTION_ALIAS_MAP = {}
PART_MAPPING_PATH = None
ACTION_MAPPING_PATH = None
MASK_CACHE = {}


def normalize_label(value):
    return str(value or "").strip().lower()


def load_alias_map(mapping_path):
    if not mapping_path:
        return {}
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"mapping file not found: {mapping_path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
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


def get_aff_point(aff):
    return aff.get("point", aff.get("affordance_point"))


def get_part_affordances(part):
    return part.get("affordances", []) or []


def get_object_affordances(obj):
    # New schema: object->parts->affordances
    part_level = []
    for part in obj.get("parts", []) or []:
        for aff in get_part_affordances(part):
            aff_item = dict(aff)
            aff_item.setdefault("part_name", part.get("part_name"))
            aff_item.setdefault("part_bbox", part.get("bbox"))
            part_level.append(aff_item)
    if part_level:
        return part_level
    # Backward compatibility: object-level affordances
    return obj.get("affordances", []) or []


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


def get_aff_bbox(aff, fallback_bbox=None):
    bbox = aff.get("affordance_bbox", aff.get("bbox"))
    if bbox is None:
        bbox = fallback_bbox
    return bbox


def point_in_mask(point, mask_path):
    if point is None or not mask_path:
        return None
    if mask_path in MASK_CACHE:
        mask = MASK_CACHE[mask_path]
    else:
        if not os.path.exists(mask_path):
            MASK_CACHE[mask_path] = None
            return None
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


def is_point_in_aff_region(point, gt_aff, fallback_bbox=None):
    mask_path = gt_aff.get("mask_path")
    if mask_path:
        in_mask = point_in_mask(point, mask_path)
        if in_mask is not None:
            return in_mask
    return is_point_in_bbox(point, get_aff_bbox(gt_aff, fallback_bbox=fallback_bbox))


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


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def harmonic_mean(a, b, eps=1e-12):
    return (2 * a * b) / (a + b + eps)


def match_objects(gt_objects, pred_objects, thr):
    matches = []
    used_pred = set()
    for gi, gt in enumerate(gt_objects):
        best_j = -1
        best_iou = 0.0
        for pj, pred in enumerate(pred_objects):
            if pj in used_pred:
                continue
            if pred.get("name") != gt.get("name"):
                continue
            score = iou(gt.get("bbox"), pred.get("bbox"))
            if score >= thr and score > best_iou:
                best_iou = score
                best_j = pj
        if best_j != -1:
            used_pred.add(best_j)
            matches.append((gi, best_j))
    return matches


def eval_parts(gt_parts, pred_parts, thr):
    matched = 0
    used_pred = set()
    for gt in gt_parts:
        best_j = -1
        best_iou = 0.0
        for j, pred in enumerate(pred_parts):
            if j in used_pred:
                continue
            if canonicalize_part_name(pred.get("part_name")) != canonicalize_part_name(gt.get("part_name")):
                continue
            score = iou(gt.get("bbox"), pred.get("bbox"))
            if score >= thr and score > best_iou:
                best_iou = score
                best_j = j
        if best_j != -1:
            used_pred.add(best_j)
            matched += 1
    gt_n = len(gt_parts)
    pred_n = len(pred_parts)
    if gt_n == 0 and pred_n == 0:
        return None, None
    recall = matched / gt_n if gt_n else 0.0
    precision = matched / pred_n if pred_n else 0.0
    return recall, precision


def eval_affordances(gt_affs, pred_affs):
    matched = 0
    used_pred = set()
    for gt in gt_affs:
        best_j = -1
        for j, pred in enumerate(pred_affs):
            if j in used_pred:
                continue
            if canonicalize_action(pred.get("action")) != canonicalize_action(gt.get("action")):
                continue
            if is_point_in_bbox(get_aff_point(pred), gt.get("affordance_bbox")):
                best_j = j
                break
        if best_j != -1:
            used_pred.add(best_j)
            matched += 1
    gt_n = len(gt_affs)
    pred_n = len(pred_affs)
    if gt_n == 0 and pred_n == 0:
        return None, None
    recall = matched / gt_n if gt_n else 0.0
    precision = matched / pred_n if pred_n else 0.0
    return recall, precision


def has_any_affordance_match(gt_affs, pred_affs):
    used_pred = set()
    for gt in gt_affs:
        for j, pred in enumerate(pred_affs):
            if j in used_pred:
                continue
            if canonicalize_action(pred.get("action")) != canonicalize_action(gt.get("action")):
                continue
            if is_point_in_aff_region(
                get_aff_point(pred),
                gt,
                fallback_bbox=gt.get("part_bbox"),
            ):
                used_pred.add(j)
                return True
    return False


def match_parts_with_indices(gt_parts, pred_parts, thr):
    matched = 0
    used_pred = set()
    pairs = []
    for gi, gt in enumerate(gt_parts):
        best_j = -1
        best_iou = 0.0
        for j, pred in enumerate(pred_parts):
            if j in used_pred:
                continue
            if canonicalize_part_name(pred.get("part_name")) != canonicalize_part_name(gt.get("part_name")):
                continue
            score = iou(gt.get("bbox"), pred.get("bbox"))
            if score >= thr and score > best_iou:
                best_iou = score
                best_j = j
        if best_j != -1:
            used_pred.add(best_j)
            matched += 1
            pairs.append((gi, best_j))
    gt_n = len(gt_parts)
    pred_n = len(pred_parts)
    if gt_n == 0 and pred_n == 0:
        return None, None, pairs
    recall = matched / gt_n if gt_n else 0.0
    precision = matched / pred_n if pred_n else 0.0
    return recall, precision, pairs


def eval_affordances_in_matched_parts(gt_parts, pred_parts, part_pairs):
    gt_total = 0
    pred_total = 0
    matched = 0
    matched_has_correct = False

    for part in gt_parts:
        gt_total += len(get_part_affordances(part))
    for part in pred_parts:
        pred_total += len(get_part_affordances(part))

    used_pred = set()
    for g_idx, p_idx in part_pairs:
        gt_affs = get_part_affordances(gt_parts[g_idx])
        pred_affs = get_part_affordances(pred_parts[p_idx])
        fallback_bbox = gt_parts[g_idx].get("bbox")
        for gt_aff in gt_affs:
            best_j = -1
            for j, pred_aff in enumerate(pred_affs):
                uid = (p_idx, j)
                if uid in used_pred:
                    continue
                if canonicalize_action(pred_aff.get("action")) != canonicalize_action(gt_aff.get("action")):
                    continue
                if is_point_in_aff_region(get_aff_point(pred_aff), gt_aff, fallback_bbox=fallback_bbox):
                    best_j = j
                    break
            if best_j != -1:
                used_pred.add((p_idx, best_j))
                matched += 1
                matched_has_correct = True

    if gt_total == 0 and pred_total == 0:
        return None, None, matched_has_correct
    recall = matched / gt_total if gt_total else 0.0
    precision = matched / pred_total if pred_total else 0.0
    return recall, precision, matched_has_correct


def count_affordance_matches_in_part_pairs(gt_parts, pred_parts, part_pairs):
    matched = 0
    used_pred = set()
    for g_idx, p_idx in part_pairs:
        gt_affs = get_part_affordances(gt_parts[g_idx])
        pred_affs = get_part_affordances(pred_parts[p_idx])
        fallback_bbox = gt_parts[g_idx].get("bbox")
        for gt_aff in gt_affs:
            best_j = -1
            for j, pred_aff in enumerate(pred_affs):
                uid = (p_idx, j)
                if uid in used_pred:
                    continue
                if canonicalize_action(pred_aff.get("action")) != canonicalize_action(gt_aff.get("action")):
                    continue
                if is_point_in_aff_region(get_aff_point(pred_aff), gt_aff, fallback_bbox=fallback_bbox):
                    best_j = j
                    break
            if best_j != -1:
                used_pred.add((p_idx, best_j))
                matched += 1
    return matched


def count_affordance_matches(gt_affs, pred_affs):
    matched = 0
    used_pred = set()
    for gt_aff in gt_affs:
        best_j = -1
        for j, pred_aff in enumerate(pred_affs):
            if j in used_pred:
                continue
            if canonicalize_action(pred_aff.get("action")) != canonicalize_action(gt_aff.get("action")):
                continue
            if is_point_in_aff_region(get_aff_point(pred_aff), gt_aff, fallback_bbox=gt_aff.get("part_bbox")):
                best_j = j
                break
        if best_j != -1:
            used_pred.add(best_j)
            matched += 1
    return matched


def summarize(scores):
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = mean([s[0] for s in scores])
    r = mean([s[1] for s in scores])
    return {"precision": p, "recall": r, "f1": f1(p, r)}


def safe_mean(values):
    return mean(values) if values else 0.0


def summarize_counts(tp, pred_total, gt_total):
    fp = max(0, pred_total - tp)
    fn = max(0, gt_total - tp)
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gt_total if gt_total else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
    }


def multiset_semantic_match(gt_items, pred_items, key):
    if key == "part_name":
        gt_vals = [canonicalize_part_name(x.get(key)) for x in gt_items if x.get(key)]
        pred_vals = [canonicalize_part_name(x.get(key)) for x in pred_items if x.get(key)]
    elif key == "action":
        gt_vals = [canonicalize_action(x.get(key)) for x in gt_items if x.get(key)]
        pred_vals = [canonicalize_action(x.get(key)) for x in pred_items if x.get(key)]
    else:
        gt_vals = [x.get(key) for x in gt_items if x.get(key)]
        pred_vals = [x.get(key) for x in pred_items if x.get(key)]
    gt_counter = Counter(gt_vals)
    pred_counter = Counter(pred_vals)
    matched = sum((gt_counter & pred_counter).values())
    return matched, len(gt_vals), len(pred_vals)


def greedy_part_semantic_pairs(gt_parts, pred_parts):
    pairs = []
    used_pred = set()
    for gi, gt in enumerate(gt_parts):
        gt_name = canonicalize_part_name(gt.get("part_name"))
        if not gt_name:
            continue
        best_j = -1
        best_iou = -1.0
        for pj, pred in enumerate(pred_parts):
            if pj in used_pred:
                continue
            if canonicalize_part_name(pred.get("part_name")) != gt_name:
                continue
            score = iou(gt.get("bbox"), pred.get("bbox"))
            if score > best_iou:
                best_iou = score
                best_j = pj
        if best_j != -1:
            used_pred.add(best_j)
            pairs.append((gi, best_j))
    return pairs


def greedy_aff_action_pairs(gt_affs, pred_affs):
    pairs = []
    used_pred = set()
    for gi, gt in enumerate(gt_affs):
        gt_action = canonicalize_action(gt.get("action"))
        if not gt_action:
            continue
        best_j = -1
        for pj, pred in enumerate(pred_affs):
            if pj in used_pred:
                continue
            if canonicalize_action(pred.get("action")) != gt_action:
                continue
            best_j = pj
            break
        if best_j != -1:
            used_pred.add(best_j)
            pairs.append((gi, best_j))
    return pairs


def main():
    global CANONICAL_EVAL, PART_ALIAS_MAP, ACTION_ALIAS_MAP, PART_MAPPING_PATH, ACTION_MAPPING_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--canonical_eval", action="store_true")
    parser.add_argument("--part_mapping_path", type=str, default=None)
    parser.add_argument("--action_mapping_path", type=str, default=None)
    args = parser.parse_args()

    CANONICAL_EVAL = args.canonical_eval
    PART_MAPPING_PATH = args.part_mapping_path
    ACTION_MAPPING_PATH = args.action_mapping_path
    if CANONICAL_EVAL:
        PART_ALIAS_MAP = load_alias_map(args.part_mapping_path) if args.part_mapping_path else {}
        ACTION_ALIAS_MAP = load_alias_map(args.action_mapping_path) if args.action_mapping_path else {}

    with open(args.data_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    obj_scores_05 = []
    obj_scores_095 = []
    part_scores_05 = []
    part_scores_095 = []
    aff_scores = []
    hier_scores = []
    json_parse_success = []

    gt_objects_with_parts = 0
    gt_objects_with_affordances = 0
    pred_objects_with_parts = 0
    pred_objects_with_affordances = 0

    matched_objects_05 = 0
    matched_gt_objects_with_parts = 0
    matched_gt_objects_with_affordances = 0
    matched_pred_objects_with_parts = 0
    matched_pred_objects_with_affordances = 0
    matched_part_mismatch = 0
    matched_aff_mismatch = 0

    part_expansion_tp = 0
    part_expansion_gt_positive = 0
    part_expansion_pred_positive = 0
    aff_expansion_tp = 0
    aff_expansion_gt_positive = 0
    aff_expansion_pred_positive = 0

    part_overexpansion_count = 0
    part_overexpansion_base = 0
    part_underexpansion_count = 0
    part_underexpansion_base = 0
    aff_overexpansion_count = 0
    aff_overexpansion_base = 0
    aff_underexpansion_count = 0
    aff_underexpansion_base = 0

    part_semantic_matched = 0
    part_semantic_gt_total = 0
    part_semantic_pred_total = 0
    aff_action_matched = 0
    aff_action_gt_total = 0
    aff_action_pred_total = 0

    part_geom_correct_05 = 0
    part_geom_correct_095 = 0
    part_semantic_pair_total = 0
    aff_point_correct_given_action = 0
    aff_action_pair_total = 0
    parse_success_count = 0
    parse_success_count_conditional = 0
    loose_parse_success_count = 0
    strict_parse_success_count = 0
    parse_eligible_gt_objects_05 = 0
    hier_em_e2e_success_count_05 = 0
    obj_gt_total_05 = 0
    obj_pred_total_05 = 0
    obj_tp_total_05 = 0
    part_gt_total_05 = 0
    part_pred_total_05 = 0
    part_tp_total_05 = 0
    aff_gt_total_05 = 0
    aff_pred_total_05 = 0
    aff_tp_total_05 = 0

    gt_parts_per_matched_object = []
    pred_parts_per_matched_object = []
    gt_affs_per_matched_object = []
    pred_affs_per_matched_object = []

    for sample in samples:
        gt_objects = sample.get("gt", {}).get("objects", [])
        pred_objects = sample.get("extracted_predictions", {}).get("objects", [])
        json_parse_success.append(
            1.0
            if isinstance(sample.get("extracted_predictions"), dict)
            and "objects" in sample.get("extracted_predictions", {})
            else 0.0
        )

        gt_objects_with_parts += sum(1 for obj in gt_objects if obj.get("parts"))
        gt_objects_with_affordances += sum(1 for obj in gt_objects if get_object_affordances(obj))
        pred_objects_with_parts += sum(1 for obj in pred_objects if obj.get("parts"))
        pred_objects_with_affordances += sum(1 for obj in pred_objects if get_object_affordances(obj))
        obj_gt_total_05 += len(gt_objects)
        obj_pred_total_05 += len(pred_objects)
        part_gt_total_05 += sum(len(obj.get("parts", []) or []) for obj in gt_objects)
        part_pred_total_05 += sum(len(obj.get("parts", []) or []) for obj in pred_objects)
        aff_gt_total_05 += sum(len(get_object_affordances(obj)) for obj in gt_objects)
        aff_pred_total_05 += sum(len(get_object_affordances(obj)) for obj in pred_objects)
        parse_eligible_gt_objects_05 += sum(
            1
            for obj in gt_objects
            if bool(obj.get("parts")) or bool(get_object_affordances(obj))
        )

        for thr, collector, part_collector in [
            (0.5, obj_scores_05, part_scores_05),
            (0.95, obj_scores_095, part_scores_095),
        ]:
            obj_matches = match_objects(gt_objects, pred_objects, thr)
            gt_n = len(gt_objects)
            pred_n = len(pred_objects)
            m = len(obj_matches)
            obj_r = m / gt_n if gt_n else 1.0
            obj_p = m / pred_n if pred_n else (1.0 if gt_n == 0 else 0.0)
            collector.append((obj_p, obj_r))
            if thr == 0.5:
                obj_tp_total_05 += m

            part_prs = []
            aff_prs = []
            exact_hits = []
            for gi, pj in obj_matches:
                gt_obj = gt_objects[gi]
                pred_obj = pred_objects[pj]
                gt_parts = gt_obj.get("parts", [])
                pred_parts = pred_obj.get("parts", [])
                gt_affs = get_object_affordances(gt_obj)
                pred_affs = get_object_affordances(pred_obj)

                part_r, part_p, part_pairs = match_parts_with_indices(gt_parts, pred_parts, thr)
                if part_r is not None and part_p is not None:
                    part_prs.append((part_p, part_r))
                if thr == 0.5:
                    matched_objects_05 += 1
                    gt_has_parts = object_has_parts(gt_obj)
                    gt_has_affs = object_has_part_affordances(gt_obj)
                    pred_has_parts = object_has_parts(pred_obj)
                    pred_has_affs = object_has_part_affordances(pred_obj)
                    matched_gt_objects_with_parts += int(gt_has_parts)
                    matched_gt_objects_with_affordances += int(gt_has_affs)
                    matched_pred_objects_with_parts += int(pred_has_parts)
                    matched_pred_objects_with_affordances += int(pred_has_affs)
                    if gt_has_parts and not pred_has_parts:
                        matched_part_mismatch += 1
                    if gt_has_affs and not pred_has_affs:
                        matched_aff_mismatch += 1

                    part_expansion_gt_positive += int(gt_has_parts)
                    part_expansion_pred_positive += int(pred_has_parts)
                    part_expansion_tp += int(gt_has_parts and pred_has_parts)
                    for g_part_idx, p_part_idx in part_pairs:
                        gt_part_has_aff = bool(get_part_affordances(gt_parts[g_part_idx]))
                        pred_part_has_aff = bool(get_part_affordances(pred_parts[p_part_idx]))
                        aff_expansion_gt_positive += int(gt_part_has_aff)
                        aff_expansion_pred_positive += int(pred_part_has_aff)
                        aff_expansion_tp += int(gt_part_has_aff and pred_part_has_aff)

                    part_overexpansion_base += int(not gt_has_parts)
                    part_overexpansion_count += int((not gt_has_parts) and pred_has_parts)
                    part_underexpansion_base += int(gt_has_parts)
                    part_underexpansion_count += int(gt_has_parts and (not pred_has_parts))
                    for g_part_idx, p_part_idx in part_pairs:
                        gt_part_has_aff = bool(get_part_affordances(gt_parts[g_part_idx]))
                        pred_part_has_aff = bool(get_part_affordances(pred_parts[p_part_idx]))
                        aff_overexpansion_base += int(not gt_part_has_aff)
                        aff_overexpansion_count += int((not gt_part_has_aff) and pred_part_has_aff)
                        aff_underexpansion_base += int(gt_part_has_aff)
                        aff_underexpansion_count += int(gt_part_has_aff and (not pred_part_has_aff))

                    aff_r, aff_p, has_correct_aff_in_matched_parts = eval_affordances_in_matched_parts(
                        gt_parts, pred_parts, part_pairs
                    )
                    if aff_r is not None and aff_p is not None:
                        aff_prs.append((aff_p, aff_r))
                    has_correct_part = len(part_pairs) > 0
                    part_tp_total_05 += len(part_pairs)
                    if gt_has_parts:
                        aff_tp_total_05 += count_affordance_matches_in_part_pairs(
                            gt_parts, pred_parts, part_pairs
                        )
                    else:
                        aff_tp_total_05 += count_affordance_matches(gt_affs, pred_affs)
                    parse_success_count_conditional += int(
                        has_correct_part and has_correct_aff_in_matched_parts
                    )

                    loose_parse_success_count += int(
                        matched_object_loose_parse_success(gt_obj, pred_obj)
                    )
                    strict_parse_success_count += int(
                        matched_object_strict_parse_success(gt_obj, pred_obj)
                    )

                    gt_has_parts = object_has_parts(gt_obj)
                    gt_has_part_affs = object_has_part_affordances(gt_obj)
                    gt_has_any_affs = object_has_any_affordances(gt_obj)
                    parse_success = False
                    if gt_has_parts and gt_has_part_affs:
                        parse_success = has_correct_part and has_correct_aff_in_matched_parts
                    elif gt_has_parts:
                        parse_success = has_correct_part
                    elif gt_has_any_affs:
                        parse_success = has_any_affordance_match(gt_affs, pred_affs)
                    parse_success_count += int(parse_success)
                    strict_success = False
                    if gt_has_parts and gt_has_part_affs:
                        strict_success = (
                            part_r == 1.0 and part_p == 1.0 and aff_r == 1.0 and aff_p == 1.0
                        )
                    elif gt_has_parts:
                        strict_success = (part_r == 1.0 and part_p == 1.0)
                    elif gt_has_any_affs:
                        root_aff_matches = count_affordance_matches(gt_affs, pred_affs)
                        strict_success = (
                            root_aff_matches == len(gt_affs)
                            and root_aff_matches == len(pred_affs)
                        )
                    hier_em_e2e_success_count_05 += int(strict_success)
                    exact_hits.append(
                        1.0
                        if part_r == 1.0 and part_p == 1.0 and aff_r == 1.0 and aff_p == 1.0
                        else 0.0
                    )

                    matched_names, gt_name_n, pred_name_n = multiset_semantic_match(
                        gt_parts, pred_parts, "part_name"
                    )
                    part_semantic_matched += matched_names
                    part_semantic_gt_total += gt_name_n
                    part_semantic_pred_total += pred_name_n

                    matched_actions, gt_act_n, pred_act_n = multiset_semantic_match(
                        gt_affs, pred_affs, "action"
                    )
                    aff_action_matched += matched_actions
                    aff_action_gt_total += gt_act_n
                    aff_action_pred_total += pred_act_n

                    part_pairs = greedy_part_semantic_pairs(gt_parts, pred_parts)
                    part_semantic_pair_total += len(part_pairs)
                    part_geom_correct_05 += sum(
                        1 for g_idx, p_idx in part_pairs if iou(gt_parts[g_idx].get("bbox"), pred_parts[p_idx].get("bbox")) >= 0.5
                    )
                    part_geom_correct_095 += sum(
                        1 for g_idx, p_idx in part_pairs if iou(gt_parts[g_idx].get("bbox"), pred_parts[p_idx].get("bbox")) >= 0.95
                    )

                    aff_pairs = greedy_aff_action_pairs(gt_affs, pred_affs)
                    aff_action_pair_total += len(aff_pairs)
                    aff_point_correct_given_action += sum(
                        1
                        for g_idx, p_idx in aff_pairs
                        if is_point_in_aff_region(
                            get_aff_point(pred_affs[p_idx]),
                            gt_affs[g_idx],
                            fallback_bbox=None,
                        )
                    )

                    gt_parts_per_matched_object.append(len(gt_parts))
                    pred_parts_per_matched_object.append(len(pred_parts))
                    gt_affs_per_matched_object.append(len(gt_affs))
                    pred_affs_per_matched_object.append(len(pred_affs))

            if part_prs:
                part_collector.append((mean([x[0] for x in part_prs]), mean([x[1] for x in part_prs])))
            if thr == 0.5:
                if aff_prs:
                    aff_scores.append((mean([x[0] for x in aff_prs]), mean([x[1] for x in aff_prs])))
                    hier_scores.append(mean(exact_hits) if exact_hits else 0.0)

    part_semantic_precision = (
        part_semantic_matched / part_semantic_pred_total if part_semantic_pred_total else 0.0
    )
    part_semantic_recall = (
        part_semantic_matched / part_semantic_gt_total if part_semantic_gt_total else 0.0
    )
    aff_action_precision = (
        aff_action_matched / aff_action_pred_total if aff_action_pred_total else 0.0
    )
    aff_action_recall = (
        aff_action_matched / aff_action_gt_total if aff_action_gt_total else 0.0
    )

    avg_gt_parts = safe_mean(gt_parts_per_matched_object)
    avg_pred_parts = safe_mean(pred_parts_per_matched_object)
    avg_gt_affs = safe_mean(gt_affs_per_matched_object)
    avg_pred_affs = safe_mean(pred_affs_per_matched_object)

    part_expansion_precision = (
        part_expansion_tp / part_expansion_pred_positive if part_expansion_pred_positive else 0.0
    )
    part_expansion_recall = (
        part_expansion_tp / part_expansion_gt_positive if part_expansion_gt_positive else 0.0
    )
    aff_expansion_precision = (
        aff_expansion_tp / aff_expansion_pred_positive if aff_expansion_pred_positive else 0.0
    )
    aff_expansion_recall = (
        aff_expansion_tp / aff_expansion_gt_positive if aff_expansion_gt_positive else 0.0
    )

    part_overexpansion_rate = (
        part_overexpansion_count / part_overexpansion_base if part_overexpansion_base else 0.0
    )
    part_underexpansion_rate = (
        part_underexpansion_count / part_underexpansion_base if part_underexpansion_base else 0.0
    )
    aff_overexpansion_rate = (
        aff_overexpansion_count / aff_overexpansion_base if aff_overexpansion_base else 0.0
    )
    aff_underexpansion_rate = (
        aff_underexpansion_count / aff_underexpansion_base if aff_underexpansion_base else 0.0
    )

    obj_summary_05 = summarize(obj_scores_05)
    obj_summary_095 = summarize(obj_scores_095)
    part_summary_05 = summarize(part_scores_05)
    part_summary_095 = summarize(part_scores_095)
    aff_summary = summarize(aff_scores)

    f_obj = obj_summary_05["f1"]
    f_part = part_summary_05["f1"]
    f_aff = aff_summary["f1"]
    eps = 1e-12
    f_hier_h = 3.0 / (
        (1.0 / (f_obj + eps))
        + (1.0 / (f_part + eps))
        + (1.0 / (f_aff + eps))
    )
    object_e2e_05 = summarize_counts(obj_tp_total_05, obj_pred_total_05, obj_gt_total_05)
    part_e2e_05 = summarize_counts(part_tp_total_05, part_pred_total_05, part_gt_total_05)
    aff_e2e_05 = summarize_counts(aff_tp_total_05, aff_pred_total_05, aff_gt_total_05)
    f_hier_e2e = 3.0 / (
        (1.0 / (object_e2e_05["f1"] + eps))
        + (1.0 / (part_e2e_05["f1"] + eps))
        + (1.0 / (aff_e2e_05["f1"] + eps))
    )

    results = {
        "object_0.5": obj_summary_05,
        "object_0.95": obj_summary_095,
        "part_0.5": part_summary_05,
        "part_0.95": part_summary_095,
        "affordance_point_in_bbox": aff_summary,
        "f_hier": {
            "f_obj": f_obj,
            "f_part_obj": f_part,
            "f_aff_part": f_aff,
            "f_hier_h": f_hier_h,
            "f_hier": f_hier_h,
        },
        "L1": object_e2e_05,
        "L2": part_e2e_05,
        "L3": aff_e2e_05,
        "hier_f1_e2e_0.5": f_hier_e2e,
        "ParseRate": (
            (loose_parse_success_count / parse_eligible_gt_objects_05)
            if parse_eligible_gt_objects_05
            else 0.0
        ),
        "strict_parse_rate_0.5": (
            (strict_parse_success_count / parse_eligible_gt_objects_05)
            if parse_eligible_gt_objects_05
            else 0.0
        ),
        "hier_em_e2e_0.5": (
            (hier_em_e2e_success_count_05 / parse_eligible_gt_objects_05)
            if parse_eligible_gt_objects_05
            else 0.0
        ),
        "parse_rate_0.5": (
            (parse_success_count / parse_eligible_gt_objects_05)
            if parse_eligible_gt_objects_05
            else 0.0
        ),
        "hierarchical_exact_match": mean(hier_scores) if hier_scores else 0.0,
        "part_expansion": {
            "precision": part_expansion_precision,
            "recall": part_expansion_recall,
            "f1": f1(part_expansion_precision, part_expansion_recall),
        },
        "aff_expansion": {
            "precision": aff_expansion_precision,
            "recall": aff_expansion_recall,
            "f1": f1(aff_expansion_precision, aff_expansion_recall),
        },
        "diagnostics": {
            "json_parse_success_rate": safe_mean(json_parse_success),
            "part_field_emission_rate": (
                pred_objects_with_parts / gt_objects_with_parts if gt_objects_with_parts else 0.0
            ),
            "affordance_field_emission_rate": (
                pred_objects_with_affordances / gt_objects_with_affordances if gt_objects_with_affordances else 0.0
            ),
            "matched_object_part_emission_rate": (
                matched_pred_objects_with_parts / matched_gt_objects_with_parts
                if matched_gt_objects_with_parts
                else 0.0
            ),
            "matched_object_affordance_emission_rate": (
                matched_pred_objects_with_affordances / matched_gt_objects_with_affordances
                if matched_gt_objects_with_affordances
                else 0.0
            ),
            "part_mismatch_rate": (
                matched_part_mismatch / matched_gt_objects_with_parts
                if matched_gt_objects_with_parts
                else 0.0
            ),
            "affordance_mismatch_rate": (
                matched_aff_mismatch / matched_gt_objects_with_affordances
                if matched_gt_objects_with_affordances
                else 0.0
            ),
            "matched_objects_at_0.5": matched_objects_05,
            "loose_parse_success_count_0.5": loose_parse_success_count,
            "strict_parse_success_count_0.5": strict_parse_success_count,
            "parse_success_count_0.5": parse_success_count,
            "hier_em_e2e_success_count_0.5": hier_em_e2e_success_count_05,
            "parse_eligible_gt_objects_0.5": parse_eligible_gt_objects_05,
            "parse_rate_given_matched_object_0.5": (
                (parse_success_count_conditional / matched_objects_05)
                if matched_objects_05
                else 0.0
            ),
            "part_expansion_tp": part_expansion_tp,
            "part_expansion_gt_positive": part_expansion_gt_positive,
            "part_expansion_pred_positive": part_expansion_pred_positive,
            "aff_expansion_tp": aff_expansion_tp,
            "aff_expansion_gt_positive": aff_expansion_gt_positive,
            "aff_expansion_pred_positive": aff_expansion_pred_positive,
            "part_overexpansion_rate": part_overexpansion_rate,
            "part_underexpansion_rate": part_underexpansion_rate,
            "aff_overexpansion_rate": aff_overexpansion_rate,
            "aff_underexpansion_rate": aff_underexpansion_rate,
            "part_semantic_precision": part_semantic_precision,
            "part_semantic_recall": part_semantic_recall,
            "part_semantic_f1": f1(part_semantic_precision, part_semantic_recall),
            "aff_action_precision": aff_action_precision,
            "aff_action_recall": aff_action_recall,
            "aff_action_f1": f1(aff_action_precision, aff_action_recall),
            "part_box_acc_given_semantic_0.5": (
                part_geom_correct_05 / part_semantic_pair_total if part_semantic_pair_total else 0.0
            ),
            "part_box_acc_given_semantic_0.95": (
                part_geom_correct_095 / part_semantic_pair_total if part_semantic_pair_total else 0.0
            ),
            "aff_point_acc_given_action": (
                aff_point_correct_given_action / aff_action_pair_total if aff_action_pair_total else 0.0
            ),
            "avg_gt_parts_per_matched_object": avg_gt_parts,
            "avg_pred_parts_per_matched_object": avg_pred_parts,
            "part_overprediction_ratio": (
                avg_pred_parts / avg_gt_parts if avg_gt_parts > 0 else 0.0
            ),
            "avg_gt_affs_per_matched_object": avg_gt_affs,
            "avg_pred_affs_per_matched_object": avg_pred_affs,
            "aff_overprediction_ratio": (
                avg_pred_affs / avg_gt_affs if avg_gt_affs > 0 else 0.0
            ),
            "canonical_eval_enabled": CANONICAL_EVAL,
            "part_mapping_path": PART_MAPPING_PATH,
            "action_mapping_path": ACTION_MAPPING_PATH,
        },
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
