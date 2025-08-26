import json
import os
from typing import Callable
import shutil
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import imageio as iio
import numpy as np
from PIL import Image, ImageSequence


@dataclass
class StepLog:
    step: int
    map_text: str
    player_action: Dict[str, Any]
    automated_actions: List[Dict[str, Any]]
    agents_state: Dict[str, Dict[str, Any]]
    scoreboard: Dict[str, int]


class ExperimentRecorder:
    def __init__(self, outdir: str, run_name: Optional[str] = None) -> None:
        # Attempt to ensure provider keys are loaded when running experiments programmatically
        try:
            from empathy.core.env import load_provider_keys  # local import to avoid cycles
            load_provider_keys()
        except Exception:
            print("Warning: provider keys not loaded")
        self.outdir = outdir
        self.run_name = run_name or time.strftime("%Y%m%d_%H%M%S")
        self.steps: List[StepLog] = []
        self.config: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {"created_at": int(time.time())}
        os.makedirs(self.run_directory, exist_ok=True)

    @property
    def run_directory(self) -> str:
        return os.path.join(self.outdir, self.run_name)

    def set_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        # Normalize and persist an optional human-readable override for the objective description
        # If provided, it will be used when generating the GIF instead of the game's default objective.
        if "objective_description" in self.config and self.config["objective_description"] is None:
            # remove empty placeholder
            self.config.pop("objective_description", None)

    def add_step(self, step_log: StepLog) -> None:
        self.steps.append(step_log)

    def json_path(self) -> str:
        return os.path.join(self.run_directory, "experiment.json")

    def html_path(self) -> str:
        return os.path.join(self.run_directory, "index.html")

    def gif_path(self) -> str:
        return os.path.join(self.run_directory, "run.gif")
    
    def video_path(self) -> str:
        return os.path.join(self.run_directory, "run.mp4")
    
    def convert_gif_to_mp4(self, gif_path: str, mp4_path: str) -> str:
        """Convert an animated GIF to MP4 while preserving dimensions and timing.

        If frame durations vary, frames are repeated to approximate timing at a
        stable FPS derived from the median frame duration.
        """
        if not os.path.exists(gif_path):
            raise FileNotFoundError(f"GIF not found: {gif_path}")
        os.makedirs(os.path.dirname(mp4_path), exist_ok=True)

        img = Image.open(gif_path)
        frames_rgb: List[Image.Image] = []
        durations_ms: List[int] = []
        for frame in ImageSequence.Iterator(img):
            # duration in ms; default to 100ms if missing
            durations_ms.append(int(frame.info.get("duration", 100)))
            frames_rgb.append(frame.convert("RGB"))

        if not frames_rgb:
            raise RuntimeError("GIF has no frames to convert")

        # Derive a stable FPS from median duration
        base_ms = max(1, int(round(float(np.median(durations_ms)))))
        fps = 1000.0 / float(base_ms)

        with iio.get_writer(mp4_path, fps=fps, codec="libx264", macro_block_size=1) as writer:
            for img_rgb, d_ms in zip(frames_rgb, durations_ms):
                repeats = max(1, int(round(float(d_ms) / float(base_ms))))
                arr = np.asarray(img_rgb)
                for _ in range(repeats):
                    writer.append_data(arr)
        return mp4_path

    def save_video_from_gif(self) -> str:
        """Convert the recorder's run.gif into run.mp4 and return the MP4 path."""
        gif_path = self.gif_path()
        mp4_path = self.video_path()
        return self.convert_gif_to_mp4(gif_path, mp4_path)
    
    def copy_assets(self) -> None:
        """Copy game assets to the experiment directory"""
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        target_assets_dir = os.path.join(self.run_directory, "assets")
        
        if os.path.exists(assets_dir):
            os.makedirs(target_assets_dir, exist_ok=True)
            for asset_file in ["ai.png", "player.png", "wall.png", "grass.png"]:
                src_path = os.path.join(assets_dir, asset_file)
                dst_path = os.path.join(target_assets_dir, asset_file)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)

    def save_json(self) -> str:
        payload = {
            "meta": self.meta,
            "config": self.config,
            "steps": [asdict(s) for s in self.steps],
        }
        path = self.json_path()
        os.makedirs(self.run_directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def save_html_viewer(self) -> str:
        # Enhanced viewer with visual grid using assets
        self.copy_assets()  # Copy assets before generating HTML
        
        html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Empathy Experiment Viewer</title>
<style>
* { box-sizing: border-box; }
body { 
  margin: 0; 
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #333;
}

.header { 
  padding: 20px 24px; 
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  display: flex; 
  align-items: center; 
  gap: 20px;
  position: sticky;
  top: 0;
  z-index: 100;
}

.container { 
  display: grid; 
  grid-template-columns: 1.2fr 0.8fr; 
  gap: 24px;
  padding: 24px;
  min-height: calc(100vh - 100px);
}

.left, .right { 
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.game-grid {
  display: grid;
  grid-template-columns: repeat(7, 80px);
  grid-template-rows: repeat(5, 80px);
  gap: 3px;
  margin: 20px 0;
  justify-content: center;
  background: #2a3f5f;
  padding: 20px;
  border-radius: 12px;
  box-shadow: inset 0 4px 12px rgba(0, 0, 0, 0.3);
}

.grid-cell {
  width: 80px;
  height: 80px;
  position: relative;
  border-radius: 6px;
  overflow: hidden;
  transition: all 0.2s ease;
}

.grid-cell:hover {
  transform: scale(1.05);
  z-index: 10;
}

.cell-bg {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.agent {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 70px;
  height: 70px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 22px;
  color: white;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.7);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  z-index: 5;
}

.agent.A { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
.agent.B { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
.agent.C { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
.agent.D { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); }

.controls { 
  display: flex; 
  align-items: center; 
  gap: 12px;
}

.btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  color: white;
  padding: 12px 20px;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.btn:active {
  transform: translateY(0);
}

.step-input {
  padding: 10px 16px;
  border: 2px solid #e1e8ed;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  text-align: center;
  width: 80px;
  transition: border-color 0.3s ease;
}

.step-input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.section-title {
  font-size: 20px;
  font-weight: 700;
  margin: 0 0 16px 0;
  color: #2d3748;
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-title::before {
  content: '';
  width: 4px;
  height: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 2px;
}

.info-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 16px;
  margin: 12px 0;
}



.action-card {
  background: #f0f4f8;
  border: 1px solid #cbd5e0;
  border-radius: 12px;
  padding: 16px;
  margin: 12px 0;
}

.action-type {
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 8px;
}

.action-details {
  font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
  background: white;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 14px;
  border: 1px solid #e2e8f0;
}

.summary {
  font-size: 18px;
  font-weight: 600;
  color: #4a5568;
}

.auto-action {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
  font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
  font-size: 14px;
}

@media (max-width: 1200px) {
  .container {
    grid-template-columns: 1fr;
    gap: 16px;
  }
  
  .game-grid {
    grid-template-columns: repeat(7, 60px);
    grid-template-rows: repeat(5, 60px);
  }
  
  .grid-cell {
    width: 60px;
    height: 60px;
  }
  
  .agent {
    width: 50px;
    height: 50px;
    font-size: 16px;
  }
}
</style>
</head>
<body>
  <div class="header">
    <div class="controls">
      <button id="prev" class="btn">◀ Previous</button>
      <input id="step" type="number" value="0" min="0" class="step-input"/> 
      <span style="font-weight: 600; color: #4a5568;">/ <span id="max">0</span></span>
      <button id="next" class="btn">Next ▶</button>
    </div>
    <div id="summary" class="summary"></div>
  </div>
  
  <div class="container">
    <div class="left">
      <h2 class="section-title">🎮 Game Environment</h2>
      <div id="gameGrid" class="game-grid"></div>
    </div>
    
    <div class="right">
      <h3 class="section-title">🎯 Player Action</h3>
      <div id="playerActionCard" class="action-card"></div>
      
      <h3 class="section-title">🤖 Automated Actions</h3>
      <div id="autoList"></div>
    </div>
  </div>

<script>
const AGENT_COLORS = {
  'A': '#4facfe',
  'B': '#43e97b', 
  'C': '#fa709a',
  'D': '#a8edea'
};

async function load() {
  const res = await fetch('experiment.json');
  const data = await res.json();
  const steps = data.steps || [];
  const stepInput = document.getElementById('step');
  const maxSpan = document.getElementById('max');
  const prevBtn = document.getElementById('prev');
  const nextBtn = document.getElementById('next');
  const gameGrid = document.getElementById('gameGrid');
  const playerActionCard = document.getElementById('playerActionCard');
  const autoList = document.getElementById('autoList');
  const summary = document.getElementById('summary');

  maxSpan.textContent = steps.length ? steps[steps.length-1].step : 0;

  function createGrid(agentPositions) {
    gameGrid.innerHTML = '';
    
    // Create 5x7 grid (height x width)
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 7; col++) {
        const cell = document.createElement('div');
        cell.className = 'grid-cell';
        
        const bg = document.createElement('img');
        bg.className = 'cell-bg';
        bg.src = 'assets/grass.png';
        bg.alt = 'grass';
        cell.appendChild(bg);
        
        // Check if any agent is at this position
        for (const [agentId, state] of Object.entries(agentPositions)) {
          if (state.position && state.position[0] === row && state.position[1] === col) {
            const agent = document.createElement('div');
            agent.className = `agent ${agentId}`;
            agent.textContent = agentId;
            agent.title = `Agent ${agentId}`;
            cell.appendChild(agent);
          }
        }
        
        gameGrid.appendChild(cell);
      }
    }
  }



  function createPlayerAction(action) {
    playerActionCard.innerHTML = '';
    
    const type = document.createElement('div');
    type.className = 'action-type';
    type.textContent = `Action: ${action.name}`;
    
    const details = document.createElement('div');
    details.className = 'action-details';
    details.textContent = JSON.stringify(action.params || {}, null, 2);
    
    playerActionCard.appendChild(type);
    playerActionCard.appendChild(details);
  }

  function createAutomatedActions(actions) {
    autoList.innerHTML = '';
    
    if (!actions || actions.length === 0) {
      const noActions = document.createElement('div');
      noActions.textContent = 'No automated actions this step';
      noActions.style.color = '#a0aec0';
      noActions.style.fontStyle = 'italic';
      autoList.appendChild(noActions);
      return;
    }
    
    actions.forEach(action => {
      const actionDiv = document.createElement('div');
      actionDiv.className = 'auto-action';
      actionDiv.textContent = JSON.stringify(action, null, 2);
      autoList.appendChild(actionDiv);
    });
  }

  function render(stepIndex) {
    if (stepIndex < 0) stepIndex = 0;
    if (stepIndex >= steps.length) stepIndex = steps.length - 1;
    const s = steps[stepIndex];
    if (!s) return;
    
    stepInput.value = s.step;
    summary.textContent = `Step ${s.step} of ${steps.length}`;
    
    // Create agent positions from the agents_state
    const agentPositions = {};
    for (const [agentId, state] of Object.entries(s.agents_state || {})) {
      // We need to extract positions from the map_text since it's not in agents_state
      // Parse the ASCII map to find agent positions
      const lines = s.map_text.split('\\n');
      for (let row = 0; row < lines.length; row++) {
        const line = lines[row];
        for (let col = 0; col < line.length; col++) {
          if (line[col] === agentId) {
            // Convert from map coordinates to grid coordinates
            const gridRow = Math.max(0, row - 1); // Remove border
            const gridCol = Math.max(0, Math.floor((col - 1) / 2)); // Remove border and account for spacing
            agentPositions[agentId] = { ...state, position: [gridRow, gridCol] };
            break;
          }
        }
      }
    }
    
    createGrid(agentPositions);
    createPlayerAction(s.player_action || {});
    createAutomatedActions(s.automated_actions || []);
  }

  prevBtn.onclick = () => {
    const currentStep = Number(stepInput.value);
    const newStep = Math.max(1, currentStep - 1);
    const stepIndex = steps.findIndex(s => s.step === newStep);
    if (stepIndex >= 0) render(stepIndex);
  };
  
  nextBtn.onclick = () => {
    const currentStep = Number(stepInput.value);
    const newStep = currentStep + 1;
    const stepIndex = steps.findIndex(s => s.step === newStep);
    if (stepIndex >= 0) render(stepIndex);
  };
  
  stepInput.onchange = () => {
    const targetStep = Number(stepInput.value);
    const stepIndex = steps.findIndex(s => s.step === targetStep);
    if (stepIndex >= 0) render(stepIndex);
  };
  
  // Initialize with first step
  if (steps.length > 0) {
    render(0);
  }
}

load();
</script>
</body>
</html>
"""
        path = self.html_path()
        os.makedirs(self.run_directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    # Utility to export gif given a renderer function
    def save_gif(self, render_fn: Callable[[list, str, str, str, str], str], title: str, objective: str) -> str:
        os.makedirs(self.run_directory, exist_ok=True)
        out_path = self.gif_path()
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        return render_fn(self.steps, out_path, title, objective, assets_dir)

    # Utility to export video given a renderer function
    def save_video(self, render_fn: Callable[[list, str, str, str, str], str], title: str, objective: str) -> str:
        os.makedirs(self.run_directory, exist_ok=True)
        # Prefer converting from GIF if it already exists to preserve exact dimensions/timing
        gif_path = self.gif_path()
        out_path = self.video_path()
        if os.path.exists(gif_path):
            try:
                return self.convert_gif_to_mp4(gif_path, out_path)
            except Exception:
                # Fall back to direct rendering if conversion fails
                pass
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        return render_fn(self.steps, out_path, title, objective, assets_dir)

