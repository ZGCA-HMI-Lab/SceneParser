#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path


PUBLIC_METRICS = ("L1", "L2", "L3", "ParseRate")


def metric_value(value):
    if isinstance(value, dict) and "f1" in value:
        return value["f1"]
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Extract the four SceneParser report metrics from hierarchical eval results."
    )
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    eval_path = Path(args.eval_json)
    data = json.loads(eval_path.read_text(encoding="utf-8"))

    output = {}
    for name in PUBLIC_METRICS:
        if name not in data:
            raise KeyError(f"missing metric key {name!r} in {eval_path}")
        output[name] = metric_value(data[name])

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
