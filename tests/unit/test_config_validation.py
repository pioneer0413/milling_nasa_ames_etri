import unittest

from milling_experiment_framework.core.config import load_and_resolve_config


class ConfigValidationTest(unittest.TestCase):
    def test_example_config_resolves(self):
        _, config, validation = load_and_resolve_config("configs/experiments/exp001_cnn1d_smoke.yaml")
        self.assertTrue(validation["ok"])
        self.assertTrue(config["experiment"]["experiment_id"])
        self.assertTrue(config["config_hash"])


if __name__ == "__main__":
    unittest.main()
