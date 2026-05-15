import random
from typing import List

from .task_conditioned_structured_parsing_obj_part_aff_drop_pseudo_stage1_task import (
    TaskConditionedStructuredParsingObjPartAffDropPseudoStage1TaskFn,
)
from .task_conditioned_structured_parsing_obj_part_aff_task import (
    TaskConditionedStructuredParsingObjPartAffTaskFn,
)
from .task_prompts.task_conditioned_structured_parsing_intent_depth_v1 import (
    JSON_ONLY_PREFIX,
)
from .task_prompts.task_conditioned_structured_parsing_when_available import (
    TASK_CONDITIONED_STRUCTURED_PARSING_WHEN_AVAILABLE,
)


class _TaskConditionedStructuredParsingObjPartAffWhenAvailableTaskFn(
    TaskConditionedStructuredParsingObjPartAffTaskFn
):
    """Pseudo-preserving Obj-Part-Aff task fn with when-available prompts."""

    def __init__(
        self,
        task_prompts: List[str] = None,
        image_min_pixels=None,
        image_max_pixels=None,
        scene_prompt_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            task_prompts=task_prompts,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
            scene_prompt_ratio=scene_prompt_ratio,
            **kwargs,
        )
        self.task_prompts = (
            task_prompts
            if task_prompts
            else TASK_CONDITIONED_STRUCTURED_PARSING_WHEN_AVAILABLE
        )

    def build_object_centric_prompt(self, obj_name):
        prompt = random.choice(self.task_prompts).replace("[OBJ]", obj_name)
        return JSON_ONLY_PREFIX + "\n" + prompt


class TaskConditionedStructuredParsingObjPartAffMixedTaskFn(object):
    """Stage-2/3 curriculum task fn mixing no-pseudo and pseudo targets.

    Each sample is routed to the pseudo-preserving branch with probability
    ``pseudo_ratio``; otherwise it uses the drop-pseudo branch. Both branches
    keep the same when-available prompt semantics used in Stage-1.
    """

    def __init__(
        self,
        task_prompts: List[str] = None,
        image_min_pixels=None,
        image_max_pixels=None,
        scene_prompt_ratio: float = 0.0,
        pseudo_ratio: float = 0.3,
        **kwargs,
    ):
        self.pseudo_ratio = max(0.0, min(1.0, pseudo_ratio))
        self.drop_pseudo_task_fn = TaskConditionedStructuredParsingObjPartAffDropPseudoStage1TaskFn(
            task_prompts=task_prompts,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
            scene_prompt_ratio=scene_prompt_ratio,
            **kwargs,
        )
        self.pseudo_task_fn = _TaskConditionedStructuredParsingObjPartAffWhenAvailableTaskFn(
            task_prompts=task_prompts,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
            scene_prompt_ratio=scene_prompt_ratio,
            **kwargs,
        )

    def __call__(self, example, ori_width, ori_height):
        if random.random() < self.pseudo_ratio:
            return self.pseudo_task_fn(example, ori_width, ori_height)
        return self.drop_pseudo_task_fn(example, ori_width, ori_height)
