import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

from empathy.core.prompt import list_models
from empathy.main import run_single_game
from empathy.core.env import load_provider_keys


ALL_SCENARIOS: List[str] = [
    "food_delivery",
    "the_duel_depresion",
    "the_maze",
    "the_listener",
    "the_protector",
]


def run_all(
    outdir: str,
    max_steps: Optional[int],
    save_html: bool,
    save_gif: bool,
    save_video: bool,
    print_prompt: bool,
    providers_models: Dict[str, List[str]],
    scenarios: List[str],
    write_csv: bool,
    seed: Optional[int],
) -> None:
    results_csv = os.path.join(outdir, "results.csv")
    rows: List[Tuple[str, str, str, str, str, str, str, str, str]] = []

    def _has_completed_run(outdir_root: str, model_name: str, scenario_name: str) -> bool:
        safe_model = model_name.replace("/", "_")
        scenario_dir = os.path.join(outdir_root, safe_model, scenario_name)
        if not os.path.isdir(scenario_dir):
            return False
        try:
            for entry in os.listdir(scenario_dir):
                run_dir = os.path.join(scenario_dir, entry)
                if os.path.isdir(run_dir):
                    if os.path.isfile(os.path.join(run_dir, "experiment.json")):
                        return True
        except Exception:
            return False
        return False
    for provider, models in providers_models.items():
        for model in models:
            for scenario in scenarios:
                # Skip if there is already a completed run for this model/scenario
                if _has_completed_run(outdir, model, scenario):
                    print(f"\n=== Skipping (already completed) scenario={scenario} provider={provider} model={model} ===")
                    continue
                print(f"\n=== Running scenario={scenario} provider={provider} model={model} ===")
                try:
                    res = run_single_game(
                        game=scenario,
                        max_steps=max_steps,
                        use_llm=True,
                        outdir_root=outdir,
                        run_name=None,
                        print_prompt=print_prompt,
                        save_html=save_html,
                        save_gif=save_gif,
                        save_video=save_video,
                        provider=provider,
                        model=model,
                        seed=seed,
                    )
                    history_json = json.dumps(res["history"], ensure_ascii=False)
                    initial_choice_value = str(res.get("initial_choice", ""))
                    initial_raw = res.get("initial_choice_raw", {}) or {}
                    if not isinstance(initial_raw, dict):
                        try:
                            initial_raw = {"raw": str(initial_raw)}
                        except Exception:
                            initial_raw = {"raw": ""}
                    initial_justification = str(initial_raw.get("rationale", initial_raw.get("raw", "")))
                    final_justification = str(res.get("evaluation", {}).get("reason", res.get("evaluation", {}).get("raw", "")))
                    row: Tuple[str, str, str, str, str, str, str, str, str] = (
                        scenario,
                        str(res.get("provider", "")),
                        res["model"],
                        initial_choice_value,
                        initial_justification,
                        str(res.get("expected_result", "")),
                        history_json,
                        str(res.get("evaluation_score", "")),
                        final_justification,
                    )
                    if write_csv:
                        try:
                            from empathy.main import _append_results_csv
                            _append_results_csv(results_csv, [row])
                            print(f"Appended CSV row to: {results_csv}")
                        except Exception as ee:
                            print(f"Failed to write CSV row: {ee}")
                    else:
                        rows.append(row)
                except Exception as e:
                    print(f"Run failed for {scenario} {provider}/{model}: {e}")
    # Fallback: if not writing incrementally (flag off) but rows were collected, dump once
    if (not write_csv) and rows:
        from empathy.main import _append_results_csv
        _append_results_csv(results_csv, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Empathy Benchmark - Batch Runner")
    parser.add_argument("--outdir", default="runs", help="Directory to store experiment runs")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-html", action="store_true")
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--scenarios", default=",".join(ALL_SCENARIOS), help="Comma-separated scenario ids to run")
    parser.add_argument("--providers", default=None, help="Comma-separated providers to include (default: all)")
    parser.add_argument("--write-csv", action="store_true", help="Append summary rows to <outdir>/results.csv like main")
    parser.add_argument("--seed", type=int, default=None, help="Random seed to pass to LLM providers")
    args = parser.parse_args()

    # Auto-load provider keys from .PROVIDER_KEYS if present (same as main)
    try:
        load_provider_keys()
    except Exception:
        print("Warning: provider keys not loaded")

    registry = list_models()
    if args.providers:
        selected = {p: registry.get(p, []) for p in [p.strip() for p in args.providers.split(",") if p.strip()]}
    else:
        selected = registry
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    run_all(
        outdir=args.outdir,
        max_steps=args.max_steps,
        save_html=bool(args.save_html),
        save_gif=bool(args.save_gif),
        save_video=bool(args.save_video),
        print_prompt=bool(args.print_prompt),
        providers_models=selected,
        scenarios=scenarios,
        write_csv=bool(args.write_csv),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()



