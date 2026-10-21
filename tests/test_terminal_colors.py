"""Tests for terminal color detection and ANSI output."""

import os
import unittest
from unittest.mock import patch

from terminal_colors import colorize


class TerminalColorTests(unittest.TestCase):
    """Verify colors follow terminal capability and user preferences."""

    def test_colorize_returns_plain_text_when_stdout_is_not_a_terminal(self):
        """Avoid ANSI escapes in redirected output and captured tests."""
        with (
            patch("terminal_colors.sys.stdout") as stdout,
            patch.dict(os.environ, {}, clear=True),
        ):
            stdout.isatty.return_value = False
            self.assertEqual(colorize("Ready", "success"), "Ready")

    def test_colorize_uses_tango_rgb_when_terminal_supports_color(self):
        """Use the GNOME Tango palette only for an interactive terminal."""
        with (
            patch("terminal_colors.sys.stdout") as stdout,
            patch.dict(os.environ, {}, clear=True),
        ):
            stdout.isatty.return_value = True
            self.assertEqual(
                colorize("Ready", "success"),
                "\x1b[38;2;138;226;52mReady\x1b[0m",
            )

    def test_colorize_respects_no_color_even_in_a_terminal(self):
        """Honor the standard NO_COLOR opt-out."""
        with (
            patch("terminal_colors.sys.stdout") as stdout,
            patch.dict(os.environ, {"NO_COLOR": ""}, clear=True),
        ):
            stdout.isatty.return_value = True
            self.assertEqual(colorize("Ready", "success"), "Ready")

    def test_colorize_respects_dumb_terminal_setting(self):
        """Do not emit ANSI codes for terminals marked as dumb."""
        with (
            patch("terminal_colors.sys.stdout") as stdout,
            patch.dict(os.environ, {"TERM": "dumb"}, clear=True),
        ):
            stdout.isatty.return_value = True
            self.assertEqual(colorize("Ready", "success"), "Ready")
