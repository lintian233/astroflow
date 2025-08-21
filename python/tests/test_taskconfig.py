import unittest
import os
import yaml


from astroflow.config.taskconfig import TaskConfig


class TestTaskConfig(unittest.TestCase):
    def setUp(self):
        """Set up a temporary config file for tests."""
        self.config_data = {
            "input": "/path/to/input",
            "output": "/path/to/output",
            "timedownfactor": 1,
            "confidence": 0.2,
            "tsample": [
                {"name": "t1", "t": 0.5},
                {"name": "t2", "t": 0.1},
            ],
            "dm_limt": [
                {"name": "limt1", "dm_low": 200, "dm_high": 250},
                {"name": "limt2", "dm_low": 300, "dm_high": 400},
            ],
            "preprocess": [
                {"clip": 0.01},
                {"meadianbulr": "1 3"},
                {"guassion": "1 5"},
            ],
            "dmrange": [
                {"name": "dm1", "dm_low": 0, "dm_high": 100, "dm_step": 0.1},
                {"name": "dm2", "dm_low": 100, "dm_high": 300, "dm_step": 0.5},
            ],
            "freqrange": [
                {"name": "freq1", "freq_start": 1000, "freq_end": 1200},
                {"name": "freq2", "freq_start": 1200, "freq_end": 1400},
            ],
        }
        self.config_file = "/tmp/test_config.yaml"
        with open(self.config_file, "w") as f:
            yaml.dump(self.config_data, f)

        # Reset singleton for clean test
        TaskConfig._instance = None
        self.task_config = TaskConfig(self.config_file)

    def tearDown(self):
        """Remove the temporary config file."""
        os.remove(self.config_file)
        TaskConfig._instance = None

    def test_singleton(self):
        """Test that TaskConfig is a singleton."""
        config1 = TaskConfig(self.config_file)
        config2 = TaskConfig(self.config_file)
        self.assertIs(config1, config2)

    def test_inputdir(self):
        self.assertEqual(self.task_config.input, self.config_data["input"])

    def test_outputdir(self):
        self.assertEqual(self.task_config.output, self.config_data["output"])

    def test_timedownfactor(self):
        self.assertEqual(
            self.task_config.timedownfactor, self.config_data["timedownfactor"]
        )

    def test_confidence(self):
        self.assertEqual(self.task_config.confidence, self.config_data["confidence"])

    def test_tsample(self):
        self.assertEqual(self.task_config.tsample, self.config_data["tsample"])

    def test_dm_limt(self):
        self.assertEqual(self.task_config.dm_limt, self.config_data["dm_limt"])

    def test_dmrange(self):
        self.assertEqual(self.task_config.dmrange, self.config_data["dmrange"])

    def test_freqrange(self):
        self.assertEqual(self.task_config.freqrange, self.config_data["freqrange"])

    def test_file_not_found(self):
        """Test FileNotFoundError for non-existent config file."""
        TaskConfig._instance = None
        with self.assertRaises(FileNotFoundError):
            TaskConfig("/tmp/non_existent_file.yaml")
