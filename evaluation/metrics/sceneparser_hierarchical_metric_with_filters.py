#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def norm(v):
    return str(v or "").strip().lower()


def is_pseudo_part(part: dict) -> bool:
    if not isinstance(part, dict):
        return False
    if part.get("is_pseudo_part", False):
        return True
    return norm(part.get("part_name")) == "__placeholder_part__"


def is_none_action(aff: dict) -> bool:
    return norm(aff.get("action")) in {"__none__", "__placeholder_action__"}


def filter_part(part: dict, ignore_none_action: bool) -> dict:
    p = dict(part)
    affs = p.get("affordances", []) or []
    if ignore_none_action:
        affs = [a for a in affs if not is_none_action(a)]
    p["affordances"] = affs
    return p


def filter_object(obj: dict, ignore_pseudo_part: bool, ignore_none_action: bool) -> dict:
    o = dict(obj)
    parts = o.get("parts", []) or []
    out_parts = []
    for part in parts:
        if ignore_pseudo_part and is_pseudo_part(part):
            continue
        out_parts.append(filter_part(part, ignore_none_action))
    o["parts"] = out_parts

    # backward compatibility: object-level affordances
    if "affordances" in o:
        obj_affs = o.get("affordances", []) or []
        if ignore_none_action:
            obj_affs = [a for a in obj_affs if not is_none_action(a)]
        o["affordances"] = obj_affs
    return o


def filter_sample(sample: dict, ignore_pseudo_part: bool, ignore_none_action: bool) -> dict:
    s = dict(sample)
    gt = dict(s.get("gt", {}) or {})
    pred = dict(s.get("extracted_predictions", {}) or {})

    gt_objs = gt.get("objects", []) or []
    pred_objs = pred.get("objects", []) or []

    gt["objects"] = [filter_object(o, ignore_pseudo_part, ignore_none_action) for o in gt_objs]
    pred["objects"] = [filter_object(o, ignore_pseudo_part, ignore_none_action) for o in pred_objs]

    s["gt"] = gt
    s["extracted_predictions"] = pred
    return s


def main():
    parser = argparse.ArgumentParser(
        description="Run sceneparser hierarchical metric with optional filtering for fair comparison."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--canonical_eval", action="store_true")
    parser.add_argument("--part_mapping_path", type=str, default=None)
    parser.add_argument("--action_mapping_path", type=str, default=None)
    parser.add_argument("--ignore_pseudo_part", action="store_true")
    parser.add_argument("--ignore_none_action", action="store_true")
    parser.add_argument(
        "--base_metric_script",
        type=str,
        default=str(Path(__file__).resolve().with_name("sceneparser_hierarchical_metric.py")),
    )
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    filtered = [
        filter_sample(s, args.ignore_pseudo_part, args.ignore_none_action)
        for s in samples
    ]

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
        for s in filtered:
            tf.write(json.dumps(s, ensure_ascii=False) + "\n")
        temp_data_path = tf.name

    cmd = [
        sys.executable,
        args.base_metric_script,
        "--data_path",
        temp_data_path,
        "--output_path",
        args.output_path,
    ]
    if args.canonical_eval:
        cmd.append("--canonical_eval")
    if args.part_mapping_path:
        cmd += ["--part_mapping_path", args.part_mapping_path]
    if args.action_mapping_path:
        cmd += ["--action_mapping_path", args.action_mapping_path]

    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise SystemExit(rc)

    # Attach filter config into output json for traceability
    out_path = Path(args.output_path)
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))
        diag = dict(data.get("diagnostics", {}) or {})
        diag["ignore_pseudo_part"] = args.ignore_pseudo_part
        diag["ignore_none_action"] = args.ignore_none_action
        diag["base_metric_script"] = args.base_metric_script
        data["diagnostics"] = diag
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "output_path": args.output_path,
                "ignore_pseudo_part": args.ignore_pseudo_part,
                "ignore_none_action": args.ignore_none_action,
                "base_metric_script": args.base_metric_script,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
