import os
from pathlib import Path

from dataset import (
    ConcatDataset,
    DataCollatorForSupervisedDataset,
    TaskConditionedStructuredParsingTSVDataset,
)
from dataset.task_fns import TaskConditionedStructuredParsingObjPartAffDropPseudoStage1TaskFn

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

SCENEPARSER_ROOT = Path(__file__).resolve().parents[2]
TSV_ROOT = Path(
    os.environ.get(
        "SCENEPARSER_TSV_DIR",
        str(SCENEPARSER_ROOT / "datasets" / "train_tsv"),
    )
)

model_name_or_path = os.environ.get("MODEL_NAME_OR_PATH", "IDEA-Research/SceneParser")

sceneparser_structured_data = dict(
    type=TaskConditionedStructuredParsingTSVDataset,
    img_tsv_file=str(TSV_ROOT / "images.tsv"),
    ann_tsv_file=str(TSV_ROOT / "annotations.tsv"),
    ann_lineidx_file=str(TSV_ROOT / "annotations.tsv.lineidx"),
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    task_fn=dict(
        type=TaskConditionedStructuredParsingObjPartAffDropPseudoStage1TaskFn,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
        scene_prompt_ratio=0.0,
    ),
    dataset_name="sceneparser_curriculum_stage1_nopseudo_when_available_data",
)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[sceneparser_structured_data],
)

data_collator = dict(type=DataCollatorForSupervisedDataset)
