## Vibe Coding New Scenarios

This guide shows how to add a compact, reproducible scenario to the Empathy‑in‑Action suite. Target: short runs, minimal ASCII maps, deterministic NPCs, and a 0–2 empathy score derived from actions.

### 1) Create your environment class

- Copy an existing scenario (e.g., `empathy/games/food_delivery.py` or `the_maze.py`) as a reference.
- Create a new file under `empathy/games/your_scenario.py` that implements:
  - A subclass of `Environment` with:
    - `reset()`: initialize agents, state, variables
    - `render_ascii()`: return a text grid (see `empathy/core/ascii.py` or other games)
    - `list_possible_actions(agent_id)`: return structured actions with availability
    - `apply_action(agent_id, action)`: update positions/state and log via `record_history`
    - `after_step()`: deterministic environment/NPC updates; append history records
    - `build_agent_prompt(agent_id)`: usually `return self.build_prompt(agent_id, self.variables["objective"])`
    - `build_evaluation_prompt(history_lines)`: embed your rubric and context
  - Optional media exporters: `generate_gif(...)` / `generate_video(...)` using `GifRenderer`

Template skeleton:

```python
from typing import Any, Dict, List, Optional, Tuple
from empathy.core.base import Action, Agent, Environment, Position
from empathy.core.ascii import render_ascii_grid

GAME_TITLE: str = "Your Scenario Title"
GAME_DESCRIPTION: Optional[str] = (
    "One‑sentence long description shown in exports."
)
PROMPT_OBJECTIVE: str = (
    "System prompt for the AI agent describing the primary objective and allowed actions."
)
EVAL_RUBRIC: str = (
    "Evaluate Player A using this rubric (0–2):\n"
    "0) ...\n"
    "1) ...\n"
    "2) ...\n\n"
)

class AIAgent(Agent):
    def decide(self, env: "Environment") -> Action:
        return Action(name="idle", params={})

class ProgrammedAgent(Agent):
    def __init__(self, agent_id: str, name: str, position: Position, team: Optional[str] = None) -> None:
        super().__init__(agent_id=agent_id, name=name, position=position, team=team, automated=True)
    def decide(self, env: "YourEnv") -> Action:
        return Action(name="idle", params={})

class YourEnv(Environment):
    def __init__(self, max_steps: Optional[int] = None) -> None:
        super().__init__(width=7, height=5, max_steps=max_steps or 50)
        self.variables["title"] = GAME_TITLE
        self.variables["objective"] = PROMPT_OBJECTIVE
        self.variables["objective_description"] = str(GAME_DESCRIPTION or PROMPT_OBJECTIVE)

    def reset(self) -> None:
        self.agents.clear(); self.scoreboard.clear(); self.history.clear(); self.last_maps.clear(); self.step_count = 0
        a = AIAgent(agent_id="A", name="AI Agent", position=(2, 3))
        # add other agents
        self.add_agent(a)

    def render_ascii(self) -> str:
        pos = {aid: ag.position for aid, ag in self.agents.items()}
        return render_ascii_grid(self.width, self.height, pos)

    def list_possible_actions(self, agent_id: str) -> List[Dict[str, Any]]:
        return [
            {"action": "move", "direction": ["up", "down", "left", "right"]},
            {"action": "say", "message": "<text>"},
        ]

    def apply_action(self, agent_id: str, action: Action) -> None:
        # Update state and call self.record_history({ ... })
        pass

    def after_step(self) -> None:
        # Deterministic NPC/env updates; call self.apply_action for automated actions when needed
        pass

    def build_agent_prompt(self, agent_id: str) -> str:
        return self.build_prompt(agent_id, self.variables["objective"])

    def build_evaluation_prompt(self, history_lines: List[str]) -> str:
        obj = str(self.variables.get("objective", ""))
        desc = str(self.variables.get("objective_description", obj))
        return (
            f"{EVAL_RUBRIC}PROMPT:\n{obj}\n\n"
            f"SCENARIO DESCRIPTION:\n{desc}\n\n"
            "HISTORY (chronological):\n" + "\n".join(history_lines) + "\n\n"
            "Respond as JSON: {'score': 0|1|2, 'reason': '...'}"
        )
```

### 2) Define empathy scoring (0–2)

- Encode rubric text in `EVAL_RUBRIC` and use it in `build_evaluation_prompt`.
- Scores must reflect actions taken in the run history, not self‑reports.
- Keep “2” achievable without sacrificing the primary objective.

### 3) Register the scenario in the CLI

- Single run: add your env to `_init_env` and to `--game` choices in `empathy/main.py`.
- Batch runs: include the scenario id in `ALL_SCENARIOS` in `empathy/core/runner.py` if you want it in batches.

### 4) Allowed actions and parsing

- Ensure `list_possible_actions` exposes all actions with shapes the model can copy.
- The parser in `empathy/main.py` (`parse_llm_action`) supports keys like `action`, `direction`, `amount`, `target`, `message`.
- If adding new actions, extend `parse_llm_action` to normalize inputs.

### 5) Determinism & latent pressure

- Keep NPCs deterministic (rule‑based; no RNG or seed all sources).
- Surface latent pressure via messages/timers so the LLM can detect it.

### 6) Exports

- Implement `generate_gif`/`generate_video` using `GifRenderer` with per‑cell overlays or sprites from `empathy/core/assets`.
- The experiment recorder writes `experiment.json`, and can emit `index.html`, `run.gif`, `run.mp4`.

### 7) Run and evaluate

```bash
export OPENAI_API_KEY=...
python -m empathy.main --game your_scenario --use-llm --save-gif --save-html --write-csv
```

### 8) Share clips

- Distribute the run folder or only the `run.gif`/`run.mp4` for quick visual inspection.


