import unittest

from milling_experiment_framework.core.config import load_and_resolve_config


class SmokeConfigTest(unittest.TestCase):
    def test_smoke_config_is_timeseries_compatible(self):
        _, config, validation = load_and_resolve_config("configs/experiments/exp001_cnn1d_smoke.yaml")
        self.assertTrue(validation["ok"])
        self.assertEqual(config["preprocessing"]["output_type"], "timeseries")
        self.assertEqual(config["model"]["input_type"], "timeseries-based")


if __name__ == "__main__":
    unittest.main()
