"""Terminal text colors inspired by GNOME Terminal's Tango palette."""

import os
import sys

COLORS = {
    "accent": (114, 159, 207),
    "error": (239, 41, 41),
    "header": (52, 226, 226),
    "muted": (136, 138, 133),
    "prompt": (52, 226, 226),
    "success": (138, 226, 52),
    "text": (211, 215, 207),
    "warning": (252, 233, 79),
}


def colorize(text, style, *, bold=False):
    """Apply a GNOME Tango RGB foreground when output supports ANSI colors."""
    if (
        not sys.stdout.isatty()
        or "NO_COLOR" in os.environ
        or os.environ.get("TERM") == "dumb"
    ):
        return str(text)
    red, green, blue = COLORS[style]
    prefix = "\x1b[1m" if bold else ""
    return f"{prefix}\x1b[38;2;{red};{green};{blue}m{text}\x1b[0m"
