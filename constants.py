"""
Global constants used throughout the project
"""
from utils.tui import bg_color_24b


COLOR_3B = {
    0: "\033[40m",  # black
    1: "\033[44m",  # blue
    2: "\033[101m",  # red
    3: "\033[42m",  # green
    4: "\033[103m",  # yellow
    5: "\033[47m",  # white (for gray, best we can do)
    6: "\033[45m",  # magenta (for fuschia)
    7: "\033[43m",  # dark yellow (for orange, best we can do)
    8: "\033[46m",  # cyan (for teal)
    9: "\033[41m",  # dark red (for brown)
}

# Taken from the arcprize website
COLORS = {
    -1: bg_color_24b(85, 85, 85),    # Grey (#555555) Custom value for nothing
    0: bg_color_24b(0, 0, 0),       # Black (#000000)
    1: bg_color_24b(30, 147, 255),  # Blue (#1E93FF)
    2: bg_color_24b(249, 60, 49),   # Red (#F93C31)
    3: bg_color_24b(79, 204, 48),   # Green (#4FCC30)
    4: bg_color_24b(255, 220, 0),   # Yellow (#FFDC00)
    5: bg_color_24b(153, 153, 153), # Gray light (#999999)
    6: bg_color_24b(229, 58, 163),  # Magenta (#E53AA3)
    7: bg_color_24b(255, 133, 27),  # Orange (#FF851B)
    8: bg_color_24b(135, 216, 241), # Blue light (#87D8F1)
    9: bg_color_24b(146, 18, 49),   # Maroon (#921231)
}

BG_COLOR = bg_color_24b(85, 85, 85)


DATA = "../ARC-AGI/data"
DEBUG = True
