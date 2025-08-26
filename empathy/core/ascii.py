from typing import Dict, Tuple

Position = Tuple[int, int]


def render_ascii_grid(width: int, height: int, agent_positions: Dict[str, Position]) -> str:
    # Top border: 14 '▮' for width 7 + borders and separators in example
    # We will compute explicit layout based on width and height with inner cell separators using '|'
    # Inner width count equals width
    # Build top/bottom borders with 14 block for 7 columns, else scale proportionally
    # A generic way: border is '▮' repeated (2 + width * 2)
    border_len = 2 + width * 2
    top_bottom = "".join(["▮" for _ in range(border_len)])

    # Build a mapping from positions to symbol (single uppercase letter)
    pos_to_char: Dict[Position, str] = {}
    for ch, pos in agent_positions.items():
        pos_to_char[pos] = ch

    lines = [top_bottom]
    for r in range(height):
        row_cells = []
        for c in range(width):
            cell_char = pos_to_char.get((r, c), "_")
            row_cells.append(cell_char)
        # Join with '|' and add side walls ▮ at both ends
        line = "▮|" + "|".join(row_cells) + "|▮"
        lines.append(line)
    lines.append(top_bottom)
    return "\n".join(lines)
