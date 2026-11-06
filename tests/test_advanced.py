import pytest
from slm_lab.lora_training import TrainConfig
def test_validate_config():
    TrainConfig(model_id="demo/model",dataset_path="demo.jsonl").validate()
    with pytest.raises(ValueError):
        TrainConfig(model_id="",dataset_path="").validate()
