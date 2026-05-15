import random
from typing import List

from .task_conditioned_structured_parsing_obj_part_aff_drop_pseudo_task import (
    TaskConditionedStructuredParsingObjPartAffDropPseudoTaskFn,
)
from .task_prompts.task_conditioned_structured_parsing_intent_depth_v1 import (
    JSON_ONLY_PREFIX,
)
from .task_prompts.task_conditioned_structured_parsing_when_available import (
    TASK_CONDITIONED_STRUCTURED_PARSING_WHEN_AVAILABLE,
)


class TaskConditionedStructuredParsingObjPartAffDropPseudoStage1TaskFn(
    TaskConditionedStructuredParsingObjPartAffDropPseudoTaskFn
):
    """Stage-1 curriculum task fn (no-pseudo + when-available prompts).

    This class keeps the same drop-pseudo data filtering behavior as
    TaskConditionedStructuredParsingObjPartAffDropPseudoTaskFn, but uses a
    dedicated object-centric prompt bank to control prompt semantics.
    """

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

