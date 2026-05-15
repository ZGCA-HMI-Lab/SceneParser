import copy

from finetuning.engine.registry import BUILDER
from torch.utils.data import ConcatDataset as TorchConcatDataset


class ConcatDataset(TorchConcatDataset):

    def __init__(self, datasets, tokenizer=None, data_args=None):
        datasets_instance = []
        for cfg in datasets:
            # 处理LazyObject，需要先解析
            if hasattr(cfg, "_module") and hasattr(cfg, "_imported"):
                # 这是一个LazyObject，需要解析
                cfg = cfg.build()

            if isinstance(cfg, dict):
                # 创建字典副本，避免修改原始配置
                cfg_copy = copy.deepcopy(cfg)
                cfg_copy["data_args"] = data_args
                cfg_copy["tokenizer"] = tokenizer
                datasets_instance.append(BUILDER.build(cfg_copy))
            elif isinstance(cfg, list):
                for item in cfg:
                    # 创建字典副本，避免修改原始配置
                    item_copy = copy.deepcopy(item)
                    item_copy["data_args"] = data_args
                    item_copy["tokenizer"] = tokenizer
                    datasets_instance.append(BUILDER.build(item_copy))
            else:
                raise ValueError(f"Invalid dataset configuration: {cfg}")
        super().__init__(datasets=datasets_instance)

    def __repr__(self):
        main_str = "Dataset as a concatenation of multiple datasets. \n"
        main_str += ",\n".join([f"{repr(dataset)}" for dataset in self.datasets])
        return main_str
