from milling_experiment_framework.splits.dataset_run_wise_split import DatasetRunWiseSplit
from milling_experiment_framework.registry.split_registry import SPLIT_REGISTRY


@SPLIT_REGISTRY.register("condition_wise")
class ConditionWiseSplit(DatasetRunWiseSplit):
    pass
