import json
import random
import re
from typing import List

from .task_prompts.task_conditioned_structured_parsing import (
    TASK_CONDITIONED_STRUCTURED_PARSING,
)
from .task_prompts.task_conditioned_structured_parsing_intent_depth_v1 import (
    JSON_ONLY_PREFIX,
    SCENE_CENTRIC_PROMPTS,
)


class TaskConditionedStructuredParsingObjPartAffDropPseudoTaskFn(object):
    """Object->Part->Aff JSON task fn with pseudo removal at training time.

    Filtering rules:
    - Remove pseudo part by name: part_name == "__placeholder_part__".
    - Remove pseudo affordance by action: action == "__none__" (and empty/none/null).
    - Keep aff_interaction_init-derived parts (they are not placeholder by name).

    Output contract:
    - Object keeps keys: name, bbox, parts.
    - Part always keeps keys: part_name, bbox, affordances.
    - For object-only: parts=[].
    - For object-part-no-aff: part.affordances=[].
    """

    PLACEHOLDER_PART_NAMES = {
        "__placeholder_part__",
        "placeholder_part",
        "placeholder part",
    }
    NONE_ACTION_NAMES = {"", "__none__", "none", "null"}

    def __init__(
        self,
        task_prompts: List[str] = None,  # compatibility
        image_min_pixels=None,
        image_max_pixels=None,
        scene_prompt_ratio: float = 0.0,
        **kwargs,
    ):
        self.min_pixels = image_min_pixels
        self.max_pixels = image_max_pixels
        self.scene_prompt_ratio = max(0.0, min(1.0, scene_prompt_ratio))

    def normalize_bbox(self, bbox, ori_width, ori_height):
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(999, int(max(0.0, min(1.0, x0 / ori_width)) * 999)))
        y0 = max(0, min(999, int(max(0.0, min(1.0, y0 / ori_height)) * 999)))
        x1 = max(0, min(999, int(max(0.0, min(1.0, x1 / ori_width)) * 999)))
        y1 = max(0, min(999, int(max(0.0, min(1.0, y1 / ori_height)) * 999)))
        if x1 < x0 or y1 < y0:
            return None
        return [x0, y0, x1, y1]

    def normalize_point(self, point, ori_width, ori_height):
        if point is None:
            return None
        x, y = point
        x = max(0, min(999, int(max(0.0, min(1.0, x / ori_width)) * 999)))
        y = max(0, min(999, int(max(0.0, min(1.0, y / ori_height)) * 999)))
        return [x, y]

    def format_box_tokens(self, bbox):
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        return "".join([f"<{x0}>", f"<{y0}>", f"<{x1}>", f"<{y1}>"])

    def format_point_tokens(self, point):
        if point is None:
            return None
        x, y = point
        return "".join([f"<{x}>", f"<{y}>"])

    def _normalize_text(self, value):
        return str(value or "").strip().lower()

    def _is_placeholder_part(self, part_name):
        return self._normalize_text(part_name) in self.PLACEHOLDER_PART_NAMES

    def _is_none_action(self, action):
        return self._normalize_text(action) in self.NONE_ACTION_NAMES

    def _extract_point(self, aff):
        # Prefer explicit point; fallback to sampled_points[0] if needed.
        point = aff.get("point")
        if point is not None:
            return point
        sampled = aff.get("sampled_points")
        if isinstance(sampled, list) and len(sampled) > 0:
            p0 = sampled[0]
            if isinstance(p0, list) and len(p0) == 2:
                return p0
        return None

    def sanitize_object(self, obj, ori_width, ori_height):
        name = obj.get("name")
        bbox = self.normalize_bbox(obj.get("bbox"), ori_width, ori_height)
        if not name:
            return None

        parts = []
        seen_parts = set()
        for part in obj.get("parts", []) or []:
            part_name = part.get("part_name")
            if not part_name or self._is_placeholder_part(part_name):
                continue

            part_bbox = self.normalize_bbox(part.get("bbox"), ori_width, ori_height)
            if part_bbox is None:
                continue
            part_bbox_token = self.format_box_tokens(part_bbox)
            part_key = (self._normalize_text(part_name), part_bbox_token)
            if part_key in seen_parts:
                continue
            seen_parts.add(part_key)

            affordances = []
            seen_affordances = set()
            for aff in part.get("affordances", []) or []:
                action = aff.get("action")
                if self._is_none_action(action):
                    continue

                raw_point = self._extract_point(aff)
                point = self.normalize_point(raw_point, ori_width, ori_height)
                if action and point is not None:
                    point_token = self.format_point_tokens(point)
                    aff_key = (self._normalize_text(action), point_token)
                    if aff_key in seen_affordances:
                        continue
                    seen_affordances.add(aff_key)
                    affordances.append({"action": action, "point": point_token})

            parts.append(
                {
                    "part_name": part_name,
                    "bbox": part_bbox_token,
                    "affordances": affordances,
                }
            )

        return {
            "name": name,
            "bbox": self.format_box_tokens(bbox) if bbox is not None else None,
            "parts": parts,
        }

    def sample_target_name(self, objects):
        valid_names = sorted({obj.get("name") for obj in objects if obj.get("name")})
        if not valid_names:
            return None
        return random.choice(valid_names)

    def build_object_centric_prompt(self, obj_name):
        prompt = random.choice(TASK_CONDITIONED_STRUCTURED_PARSING).replace("[OBJ]", obj_name)
        return JSON_ONLY_PREFIX + "\n" + prompt

    def _dedup_objects(self, objects):
        deduped = []
        seen = set()
        for obj in objects:
            key = (
                self._normalize_text(obj.get("name")),
                obj.get("bbox"),
                json.dumps(obj.get("parts", []), ensure_ascii=False, sort_keys=True),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(obj)
        return deduped

    def _parse_x0_from_box_tokens(self, box_tokens):
        if not isinstance(box_tokens, str):
            return 10**9
        nums = re.findall(r"<(\d+)>", box_tokens)
        if len(nums) != 4:
            return 10**9
        return int(nums[0])

    def _sort_objects_by_x0(self, objects):
        return sorted(
            objects,
            key=lambda o: (
                self._parse_x0_from_box_tokens(o.get("bbox")),
                self._normalize_text(o.get("name")),
            ),
        )

    def build_scene_centric_prompt(self):
        return JSON_ONLY_PREFIX + "\n" + random.choice(SCENE_CENTRIC_PROMPTS)

    def _build_object_answer(self, target_name, objects, ori_width, ori_height):
        matched_objects = []
        for obj in objects:
            if obj.get("name") != target_name:
                continue
            sanitized = self.sanitize_object(obj, ori_width, ori_height)
            if sanitized is not None:
                matched_objects.append(sanitized)
        matched_objects = self._dedup_objects(matched_objects)
        matched_objects = self._sort_objects_by_x0(matched_objects)
        return json.dumps({"objects": matched_objects}, ensure_ascii=False)

    def _build_scene_answer(self, objects, ori_width, ori_height):
        all_objects = []
        for obj in objects:
            sanitized = self.sanitize_object(obj, ori_width, ori_height)
            if sanitized is not None:
                all_objects.append(sanitized)
        all_objects = self._dedup_objects(all_objects)
        all_objects = self._sort_objects_by_x0(all_objects)
        return json.dumps({"objects": all_objects}, ensure_ascii=False)

    def __call__(self, example, ori_width, ori_height):
        annotations = example["annotations"]
        objects = annotations.get("objects", [])

        is_scene_query = random.random() < self.scene_prompt_ratio

        if is_scene_query:
            question = self.build_scene_centric_prompt()
            answer = self._build_scene_answer(objects, ori_width, ori_height)
        else:
            target_name = self.sample_target_name(objects)
            query_name = target_name if target_name is not None else "target object"
            question = self.build_object_centric_prompt(query_name)
            answer = self._build_object_answer(target_name, objects, ori_width, ori_height)

        example["conversations"] = [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ]
        return example
