from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio as iio


class GifRenderer:
    """Reusable GIF renderer for grid-based games.

    Responsibilities:
    - Manage fonts, layout (grid vs. actions panel), and header with title/objective
    - Draw tiled grid using a floor texture
    - Place agent sprites by parsing ASCII maps (consistent with core.ascii rendering)
    - Provide hooks for per-cell overlays (e.g., water-source markers)

    Games should:
    - Provide sprite filenames per agent_id
    - Provide grid dimensions and optional cell overlay callback
    - Call render_gif(...) with a list of StepLog-like entries
    """

    def __init__(
        self,
        assets_dir: str,
        cell_size: int = 96,
        right_panel_width: int = 520,
        pad: int = 24,
        floor_asset: str = "floor.png",
    ) -> None:
        self.assets_dir = assets_dir
        self.cell = cell_size
        self.right_w = right_panel_width
        self.pad = pad
        self.floor_asset = floor_asset
        # Expose per-frame agent state for overlays that need dynamic info
        self.current_step_agents_state: Dict[str, Dict[str, Any]] = {}

        # Fonts
        self.title_font: ImageFont.ImageFont
        self.text_font: ImageFont.ImageFont
        self.small_font: ImageFont.ImageFont
        self.section_font: ImageFont.ImageFont
        self._load_fonts()

        # Preload floor texture
        self.floor_img = self._load_asset(self.floor_asset).resize((self.cell, self.cell), Image.NEAREST)

    # --------- Asset/Font helpers ---------
    def _find_font(self) -> Optional[str]:
        if os.path.isdir(self.assets_dir):
            for fname in os.listdir(self.assets_dir):
                if fname.lower().endswith((".ttf", ".otf")):
                    return os.path.join(self.assets_dir, fname)
        return None

    def _load_fonts(self) -> None:
        font_path = self._find_font()
        try:
            if font_path and os.path.exists(font_path):
                self.title_font = ImageFont.truetype(font_path, 36)
                self.text_font = ImageFont.truetype(font_path, 20)
                self.small_font = ImageFont.truetype(font_path, 18)
                self.section_font = ImageFont.truetype(font_path, 20)
            else:
                self.title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
                self.text_font = ImageFont.truetype("DejaVuSans.ttf", 20)
                self.small_font = ImageFont.truetype("DejaVuSans.ttf", 18)
                self.section_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        except Exception:
            self.title_font = ImageFont.load_default()
            self.text_font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()
            self.section_font = ImageFont.load_default()

    def _load_asset(self, name: str) -> Image.Image:
        path = os.path.join(self.assets_dir, name)
        if not os.path.exists(path):
            # Fallback placeholder
            img = Image.new("RGBA", (self.cell, self.cell), (200, 200, 200, 255))
            dr = ImageDraw.Draw(img)
            dr.rectangle([(0, 0), (self.cell - 1, self.cell - 1)], outline=(120, 120, 120), width=3)
            return img
        return Image.open(path).convert("RGBA")

    # --------- Text helpers ---------
    def _wrap(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> List[str]:
        words = (text or "").split()
        lines: List[str] = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if draw.textlength(test, font=font) > max_w:
                if cur:
                    lines.append(cur)
                cur = w
            else:
                cur = test
        if cur:
            lines.append(cur)
        return lines

    def _bold(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont, fill: Tuple[int, int, int, int]) -> None:
        x, y = xy
        draw.text((x, y), text, font=font, fill=fill)
        for dx, dy in ((1, 0), (0, 1), (1, 1)):
            draw.text((x + dx, y + dy), text, font=font, fill=fill)

    # --------- Core rendering ---------
    def _parse_agent_positions(self, map_text: str, agent_ids: Iterable[str]) -> Dict[str, Tuple[int, int]]:
        pos: Dict[str, Tuple[int, int]] = {}
        if not map_text:
            return pos
        ids = {aid for aid in agent_ids if isinstance(aid, str) and len(aid) == 1}
        lines = map_text.split("\n")
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if ch in ids:
                    grid_r = max(0, r - 1)
                    grid_c = max(0, (c - 1) // 2)
                    pos[ch] = (grid_r, grid_c)
        return pos

    def _render_frames(
        self,
        steps: List[Any],
        title: str,
        objective: str,
        grid_cols: int,
        grid_rows: int,
        sprite_files: Dict[str, str],
        per_cell_overlay: Optional[Callable[[Image.Image, int, int, int, int], None]] = None,
        player_label: str = "AI Agent Actions",
        auto_label: str = "Automated Actions",
    ) -> Tuple[List[Image.Image], int, int]:

        grid_w = grid_cols * self.cell
        grid_h = grid_rows * self.cell
        header_h = 140
        canvas_w = self.pad * 3 + grid_w + self.right_w

        # Pre-wrap objective for dynamic header height
        tmp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        td = ImageDraw.Draw(tmp)
        text_x = self.pad + 24
        max_w = canvas_w - text_x - self.pad - 24
        desc_lines = self._wrap(td, objective, self.text_font, max_w)
        line_h = 26
        header_h = max(160, 70 + len(desc_lines) * line_h + 16)
        canvas_h = header_h + self.pad * 2 + grid_h

        # Load sprites
        sprites: Dict[str, Image.Image] = {}
        for aid, fname in sprite_files.items():
            try:
                sprites[aid] = self._load_asset(fname).resize((int(self.cell * 0.8), int(self.cell * 0.8)), Image.NEAREST)
            except Exception:
                sprites[aid] = Image.new("RGBA", (int(self.cell * 0.8), int(self.cell * 0.8)), (180, 180, 180, 255))

        frames: List[Image.Image] = []
        total_steps = len(steps)

        for idx, s in enumerate(steps):
            # Step access helper
            def field(name: str, default: Any) -> Any:
                v = getattr(s, name, None)
                if v is not None:
                    return v
                if isinstance(s, dict):
                    return s.get(name, default)
                return default

            map_text: str = field("map_text", "")
            player_action: Dict[str, Any] = field("player_action", {})
            auto_actions: List[Dict[str, Any]] = field("automated_actions", [])
            step_num: int = int(field("step", idx + 1))
            agents_state: Dict[str, Dict[str, Any]] = field("agents_state", {})
            # Expose current step agent state for per-cell overlays
            try:
                self.current_step_agents_state = dict(agents_state)
            except Exception:
                self.current_step_agents_state = {}

            agent_positions = self._parse_agent_positions(map_text, agents_state.keys())

            frame = Image.new("RGBA", (canvas_w, canvas_h), (240, 241, 245, 255))
            draw = ImageDraw.Draw(frame)

            # Header card
            header_box = [self.pad, self.pad, canvas_w - self.pad, header_h]
            draw.rounded_rectangle(header_box, radius=24, fill=(222, 222, 222, 255))
            draw.text((self.pad + 24, self.pad + 16), title, font=self.title_font, fill=(20, 20, 20, 255))
            y = self.pad + 64
            for i, line in enumerate(desc_lines):
                draw.text((text_x, y + i * line_h), line, font=self.text_font, fill=(30, 30, 30, 255))

            # Environment card
            env_top = header_h + self.pad
            env_box = [self.pad, env_top, self.pad + grid_w + 32, env_top + grid_h + 32]
            draw.rounded_rectangle(env_box, radius=24, fill=(230, 230, 230, 255))
            self._bold(draw, (self.pad + 24, env_top + 10), "Environment", self.section_font, (20, 20, 20, 255))
            step_text = f"Step {step_num} / {total_steps}"
            step_w = draw.textlength(step_text, font=self.text_font)
            draw.text((env_box[2] - 24 - step_w, env_top + 10), step_text, font=self.text_font, fill=(20, 20, 20, 255))

            # Draw grid
            grid_origin = (self.pad + 24, env_top + 40)
            for r in range(grid_rows):
                for c in range(grid_cols):
                    x = grid_origin[0] + c * self.cell
                    y_cell = grid_origin[1] + r * self.cell
                    frame.alpha_composite(self.floor_img, (x, y_cell))
                    if per_cell_overlay is not None:
                        per_cell_overlay(frame, r, c, x, y_cell)

            # Draw agents
            for aid, pos in agent_positions.items():
                sprite = sprites.get(aid)
                if sprite is None:
                    continue
                rr, cc = pos
                x = grid_origin[0] + cc * self.cell + (self.cell - sprite.width) // 2
                y_cell = grid_origin[1] + rr * self.cell + (self.cell - sprite.height) // 2
                frame.alpha_composite(sprite, (x, y_cell))

            # Actions card
            right_left = self.pad * 2 + grid_w + 32
            right_box = [right_left, env_top, right_left + self.right_w, env_top + grid_h + 32]
            draw.rounded_rectangle(right_box, radius=24, fill=(230, 230, 230, 255))
            self._bold(draw, (right_left + 24, env_top + 10), "Actions", self.section_font, (20, 20, 20, 255))

            y_cursor = env_top + 40
            draw.text((right_left + 24, y_cursor), player_label, font=self.text_font, fill=(40, 40, 80, 255))
            y_cursor += 26
            act_name = str(player_action.get("name", "")).strip() or "(none)"
            act_params = player_action.get("params", {})
            # Wrap player action name to avoid overflowing the right panel
            name_lines = self._wrap(draw, f"- {act_name}", self.small_font, self.right_w - 80)
            for line in name_lines:
                draw.text((right_left + 40, y_cursor), line, font=self.small_font, fill=(20, 20, 20, 255))
                y_cursor += 22
            if act_params:
                params_str = ", ".join(f"{k}={v}" for k, v in act_params.items())
                # Wrap params with a slightly deeper indent
                param_lines = self._wrap(draw, params_str, self.small_font, self.right_w - 100)
                for line in param_lines:
                    draw.text((right_left + 60, y_cursor), line, font=self.small_font, fill=(70, 70, 70, 255))
                    y_cursor += 22

            y_cursor += 8
            draw.text((right_left + 24, y_cursor), auto_label, font=self.text_font, fill=(40, 40, 80, 255))
            y_cursor += 26
            if not auto_actions:
                draw.text((right_left + 40, y_cursor), "- none", font=self.small_font, fill=(80, 80, 80, 255))
            else:
                for aa in auto_actions[:8]:
                    text = str(aa)
                    # Wrap automated action text within the right panel width
                    auto_lines = self._wrap(draw, f"- {text}", self.small_font, self.right_w - 80)
                    for line in auto_lines:
                        draw.text((right_left + 40, y_cursor), line, font=self.small_font, fill=(20, 20, 20, 255))
                        y_cursor += 22

            frames.append(frame.convert("P", palette=Image.ADAPTIVE))

        # Clear dynamic state
        self.current_step_agents_state = {}

        if not frames:
            frames = [Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))]
        return frames, canvas_w, canvas_h

    def render_gif(
        self,
        steps: List[Any],
        out_path: str,
        title: str,
        objective: str,
        grid_cols: int,
        grid_rows: int,
        sprite_files: Dict[str, str],
        per_cell_overlay: Optional[Callable[[Image.Image, int, int, int, int], None]] = None,
        player_label: str = "AI Agent Actions",
        auto_label: str = "Automated Actions",
        seconds_per_step: float = 1,
    ) -> str:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        frames, canvas_w, canvas_h = self._render_frames(
            steps=steps,
            title=title,
            objective=objective,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            sprite_files=sprite_files,
            per_cell_overlay=per_cell_overlay,
            player_label=player_label,
            auto_label=auto_label,
        )
        pal_frames = [fr.convert("P", palette=Image.ADAPTIVE) for fr in frames]
        # Convert desired seconds per step to milliseconds for GIF frame duration
        try:
            spf = max(0.05, float(seconds_per_step))
        except Exception:
            spf = 0.6
        duration_ms = int(round(spf * 1000))
        pal_frames[0].save(
            out_path,
            save_all=True,
            append_images=pal_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        return out_path

    def render_video(
        self,
        steps: List[Any],
        out_path: str,
        title: str,
        objective: str,
        grid_cols: int,
        grid_rows: int,
        sprite_files: Dict[str, str],
        per_cell_overlay: Optional[Callable[[Image.Image, int, int, int, int], None]] = None,
        player_label: str = "AI Agent Actions",
        auto_label: str = "Automated Actions",
        fps: int = 30,
        seconds_per_step: float = 2.0,
    ) -> str:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        frames, canvas_w, canvas_h = self._render_frames(
            steps=steps,
            title=title,
            objective=objective,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            sprite_files=sprite_files,
            per_cell_overlay=per_cell_overlay,
            player_label=player_label,
            auto_label=auto_label,
        )
        # Write one frame per step at an effective rate of 1/seconds_per_step FPS
        # This preserves the GIF resolution and minimizes file size by avoiding duplicates
        fps_used = max(0.1, 1.0 / max(0.001, float(seconds_per_step)))
        rgb_frames = [np.asarray(fr.convert("RGB")) for fr in frames]
        with iio.get_writer(out_path, fps=fps_used, codec="libx264", macro_block_size=1) as writer:
            for arr in rgb_frames:
                writer.append_data(arr)
        return out_path


