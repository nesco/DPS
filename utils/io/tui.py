import os

# Constants

FALLBACK_BG = {
    0: "\033[40m",  # black
    1: "\033[44m",  # blue
    2: "\033[101m",  # red
    3: "\033[42m",  # green
    4: "\033[103m",  # yellow
    5: "\033[47m",  # white (gray)
    6: "\033[45m",  # magenta
    7: "\033[43m",  # dark yellow (orange)
    8: "\033[46m",  # cyan (teal)
    9: "\033[41m",  # dark red (brown)
}

RESET = "\033[0m"


def supports_256_colors() -> bool:
    term = os.getenv("TERM", "")
    return "256color" in term


def supports_true_color() -> bool:
    """
    Return True if the terminal claims to support 24-bit (true-color).
    We check COLORTERM and TERM for the usual markers.
    """
    # 1) Check COLORTERM
    ct = os.getenv("COLORTERM", "")
    if "truecolor" in ct.lower() or "24bit" in ct.lower():
        return True

    # 2) Check TERM
    term = os.getenv("TERM", "")
    if "truecolor" in term.lower() or "24bit" in term.lower():
        return True

    return False


def bg_color_8b(code: int) -> str:
    """
    Return the ANSI escape code for background color 'code'.
    If 256-colors are supported, use 48;5;code; otherwise fall back.
    """
    return f"\033[48;5;{code}m"


def fg_color_8b(code: int) -> str:
    """
    Same for foreground (text) colors: 38;5;code
    """
    return f"\033[38;5;{code}m"


def fg_color_24b(red: int, green: int, blue: int) -> str:
    return f"\033[38;2;{red};{green};{blue}m"


def bg_color_24b(red: int, green: int, blue: int) -> str:
    return f"\033[48;2;{red};{green};{blue}m"


if __name__ == "__main__":
    for i in [1, 21, 93, 200, 254]:
        print(f"{bg_color_8b(i)} {i:3d} {RESET}", end=" ")
    print()
    for i in range(100, 200, 50):
        for j in range(100, 200, 50):
            for k in range(100, 200, 50):
                print(
                    f"{bg_color_24b(i, j, k)} {i:3d}, {j:3d}, {k:3d} {RESET}", end=" "
                )
