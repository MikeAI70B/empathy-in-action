from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


Position = Tuple[int, int]


@dataclass
class Action:
    name: str
    params: Dict[str, Any]


class Agent(abc.ABC):
    def __init__(self, agent_id: str, name: str, position: Position, team: Optional[str] = None, automated: bool = False) -> None:
        self.agent_id = agent_id
        self.name = name
        self.position = position
        # Team is optional; normalize to empty string for stable string ops
        self.team = team if team is not None else ""
        self.automated = automated
        self.state: Dict[str, Any] = {}
        self.messages: List[str] = []

    @abc.abstractmethod
    def decide(self, env: "Environment") -> Action:
        """Return an Action. Only called for automated agents.
        Human/LLM-driven agents are decided externally via prompt/CLI.
        """
        raise NotImplementedError


class Environment(abc.ABC):
    def __init__(self, width: int, height: int, max_steps: int = 50) -> None:
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.step_count = 0
        self.agents: Dict[str, Agent] = {}
        self.scoreboard: Dict[str, int] = {}
        self.history: List[Dict[str, Any]] = []
        self.last_maps: List[str] = []
        self.variables: Dict[str, Any] = {}

    # ----- Abstracts -----
    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def render_ascii(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def list_possible_actions(self, agent_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def apply_action(self, agent_id: str, action: Action) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def after_step(self) -> None:
        """Environment updates after all agent actions for a tick (automations, timers)."""
        raise NotImplementedError

    # ----- Media / Visualization -----
    @abc.abstractmethod
    def generate_gif(self, steps: "List[Any]", out_path: str, title: str, objective: str, assets_dir: str) -> str:
        """Generate an animated GIF visualizing the run.

        Implementations should render each step into a frame that includes:
        - a visual representation of the environment using the provided assets
        - a sidebar (or overlay) summarizing the player's action and notable automated actions

        Arguments:
        - steps: a list of StepLog-like objects (shape compatible with empathy.core.experiment.StepLog)
        - out_path: destination path for the animated GIF
        - title: title to use in the header
        - objective: textual objective/description to include in the header
        - assets_dir: directory containing common PNG assets

        Return the created file path.
        """
        raise NotImplementedError

    def generate_video(self, steps: "List[Any]", out_path: str, title: str, objective: str, assets_dir: str) -> str:
        """Generate an MP4 video visualizing the run.

        Default implementation is not provided for all environments.
        Specific games that support video should override this.
        """
        raise NotImplementedError("Video export is not implemented for this environment.")

    # ----- Helpers -----
    def add_agent(self, agent: Agent) -> None:
        self.agents[agent.agent_id] = agent
        self.scoreboard.setdefault(agent.agent_id, 0)

    def record_history(self, record: Dict[str, Any]) -> None:
        self.history.append(record)

    def tick(self) -> None:
        self.step_count += 1
        current_map = self.render_ascii()
        self.last_maps.append(current_map)
        if len(self.last_maps) > 4:
            self.last_maps = self.last_maps[-4:]
        self.after_step()

    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_occupied(self, pos: Position, exclude_agent_id: Optional[str] = None) -> bool:
        for aid, ag in self.agents.items():
            if exclude_agent_id is not None and aid == exclude_agent_id:
                continue
            if ag.position == pos:
                return True
        return False

    def move_agent(self, agent_id: str, delta: Tuple[int, int]) -> bool:
        agent = self.agents[agent_id]
        r, c = agent.position
        dr, dc = delta
        new_pos = (r + dr, c + dc)
        if not self.in_bounds(new_pos):
            return False
        if self.is_occupied(new_pos, exclude_agent_id=agent_id):
            return False
        agent.position = new_pos
        return True

    # ----- Metadata helpers -----
    def get_title(self) -> str:
        """Return a human-friendly game title for visuals.

        Defaults to the environment class name if no 'title' entry is set
        in `self.variables` by the specific game implementation.
        """
        return str(self.variables.get("title", self.__class__.__name__))

    def build_prompt(self, agent_id: str, objective: str) -> str:
        maps_to_show = self.last_maps[-4:] if self.last_maps else [self.render_ascii()]
        text_maps = []
        if len(maps_to_show) == 1:
            text_maps = ["CURRENT MAP:\n" + maps_to_show[-1]]
        else:
            text_maps = [
                "MAP -3:\n" + maps_to_show[-4] if len(maps_to_show) >= 4 else None,
                "MAP -2:\n" + maps_to_show[-3] if len(maps_to_show) >= 3 else None,
                "MAP -1:\n" + maps_to_show[-2] if len(maps_to_show) >= 2 else None,
                "CURRENT MAP:\n" + maps_to_show[-1],
            ]
            text_maps = [m for m in text_maps if m is not None]

        lines: List[str] = []
        lines.append(f"PROMPT: {objective}")
        lines.append("")
        lines.extend(text_maps)
        lines.append("")
        lines.append("AGENTS STATE:")
        for aid, ag in self.agents.items():
            entry = f"- {aid} ({ag.name})"
            if getattr(ag, "team", ""):
                entry += f" team={ag.team}"
            entry += f" pos={ag.position} state={ag.state}"
            lines.append(entry)
        lines.append("")
        lines.append(f"SCOREBOARD: {self.scoreboard}")
        # Filter out description/title keys from variables for prompts
        filtered_vars = {
            k: v
            for k, v in self.variables.items()
            if all(s not in str(k).lower() for s in ("description", "title"))
        }
        lines.append(f"VARS: {filtered_vars}")
        lines.append("")
        lines.append("HISTORY (from previous to current):")
        for rec in self.history[-50:]:
            lines.append(f"- step={rec.get('step')} actor={rec.get('actor')} action={rec.get('action')}")
        lines.append("")
        lines.append("POSSIBLE ACTIONS:")
        for a in self.list_possible_actions(agent_id):
            lines.append(f"- {a}")
        lines.append("")
        lines.append("Respond with a JSON object selecting exactly one action. Example:\n{" +
                     "\"action\": \"walk\", \"direction\": \"up\"}")
        return "\n".join(lines)
