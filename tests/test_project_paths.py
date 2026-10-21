"""Tests for organized assets and backward-compatible path resolution."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model_manager_training as training


class ProjectPathTests(unittest.TestCase):
    """Verify asset lookup works with new and legacy project layouts."""

    def test_asset_lookup_prefers_organized_directory(self):
        """Choose a moved asset over an identically named legacy asset."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_directory = root / "data"
            data_directory.mkdir()
            organized_asset = data_directory / "samples.json"
            organized_asset.touch()
            (root / "samples.json").touch()

            with patch.object(training, "BASE_DIR", root):
                result = training.resolve_asset_path("samples.json", data_directory)

            self.assertEqual(result, organized_asset)

    def test_asset_lookup_supports_legacy_directory(self):
        """Continue reading an asset left in the old project root."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy_asset = root / "samples.json"
            legacy_asset.touch()

            with patch.object(training, "BASE_DIR", root):
                result = training.resolve_asset_path("samples.json", root / "data")

            self.assertEqual(result, legacy_asset)

    def test_legacy_preferences_are_preserved_when_saving(self):
        """Carry old-root preferences into the new config directory."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_directory = root / "config"
            legacy_preferences = root / "training_preferences.json"
            legacy_preferences.write_text(
                json.dumps({"existing": {"batch_size": 16}}),
                encoding="utf-8",
            )

            with (
                patch.object(training, "CONFIG_DIR", config_directory),
                patch.object(
                    training,
                    "TRAINING_PREFERENCES_PATH",
                    config_directory / "training_preferences.json",
                ),
                patch.object(
                    training,
                    "LEGACY_TRAINING_PREFERENCES_PATH",
                    legacy_preferences,
                ),
            ):
                training.save_training_preferences("new", 12, 8, 0.02)

            saved_preferences = json.loads(
                (config_directory / "training_preferences.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(saved_preferences["existing"]["batch_size"], 16)
        self.assertEqual(saved_preferences["new"]["epochs"], 12)


if __name__ == "__main__":
    unittest.main()
