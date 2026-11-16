"""Interactive console menu for the Kazakh legal-act classification project.

Run from the project root:   .venv/bin/python main.py

Every menu item calls the `run()` function of one module in src/. Errors are
caught and printed so that one failing step never crashes the menu.
"""
from __future__ import annotations

import importlib
import inspect
import time
import traceback
from datetime import datetime

from src import config  # noqa: F401  (pins GPU 0 before any torch import)

# menu key -> (title, module(s), extra kwargs, asks for quick mode?)
MENU: dict[str, tuple[str, str | list[str] | None, dict, bool]] = {
    "1": ("Prepare data (download + parquet + labels)", "src.data_io", {}, False),
    "2": ("Data audit", "src.audit", {}, False),
    "3": ("Build SQLite + run SQL queries", "src.database", {}, False),
    "4": ("EDA", "src.eda", {}, False),
    "5": ("Leakage check", "src.leakage", {}, False),
    "6": ("Train baselines (TF-IDF + LogisticRegression)", "src.baseline", {}, False),
    "7": ("Train tiny transformer (own BPE, from scratch)", ["src.bpe_tokenizer", "src.tiny_transformer"],
          {}, True),
    "8": ("Fine-tune e5-small (+ frozen e5 + LR)", "src.finetune_e5", {}, True),
    "9": ("Compare models (benchmark table + figures)", "src.compare", {}, False),
    "10": ("Error analysis", "src.error_analysis", {}, False),
    "11": ("Classify your own title", None, {}, False),
    "0": ("Exit", None, {}, False),
}


def print_menu() -> None:
    """Show the numbered menu."""
    print("\n" + "=" * 64)
    print(" Tiny Transformer for Kazakh Legal Act Classification")
    print("=" * 64)
    for key, (title, _, _, _) in MENU.items():
        print(f" {key:>2}. {title}")


def ask(prompt: str, valid: set[str] | None = None, default: str | None = None) -> str:
    """input() with validation; re-asks until the answer is acceptable."""
    while True:
        try:
            answer = input(prompt).strip()
        except EOFError:                       # e.g. piped input ended
            return "0" if valid and "0" in valid else (default or "")
        if not answer and default is not None:
            return default
        if valid is None or answer in valid:
            return answer
        print(f"  Invalid choice {answer!r}. Expected one of: {', '.join(sorted(valid, key=lambda v: (len(v), v)))}")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """y/n question with a default."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = ask(prompt + suffix, {"y", "n", "yes", "no", "Y", "N"}, default="y" if default else "n")
    return answer.lower().startswith("y")


def call_step(module_name: str, kwargs: dict | None = None, quick: bool = False) -> dict | None:
    """Import a module and call its run(); `quick` is passed only if run() accepts it."""
    kwargs = dict(kwargs or {})
    start = time.perf_counter()
    try:
        module = importlib.import_module(module_name)
        params = inspect.signature(module.run).parameters
        if quick and "quick" in params:
            kwargs["quick"] = True
        result = module.run(**kwargs)
        print(f"\n[ok] {module_name}.run() finished in {time.perf_counter() - start:.1f}s")
        if isinstance(result, dict):
            for key, value in list(result.items())[:12]:
                print(f"   {key}: {value}")
        return result if isinstance(result, dict) else {"result": result}
    except ModuleNotFoundError as exc:
        print(f"[error] module not available: {exc}")
    except KeyboardInterrupt:
        print("\n[interrupted] back to the menu")
    except Exception as exc:  # keep the menu alive whatever happens inside a step
        print(f"[error] {module_name} failed: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
    return None


def classify_loop() -> None:
    """Ask for titles until an empty line; show every model's prediction for both tasks."""
    try:
        from src import predict
    except Exception as exc:
        print(f"[error] prediction module unavailable: {exc}")
        return
    print("\nType a Kazakh legal-act title (empty line returns to the menu).")
    print("Example: " + predict.EXAMPLES[0])
    history: list[dict] = []
    while True:
        try:
            text = input("\nTitle> ").strip()
        except EOFError:
            break
        if not text:
            break
        if len(text) < 3:
            print("  Please type a longer title (at least 3 characters).")
            continue
        for task in config.TASKS:
            try:
                results = predict.predict(text, task)
                print(predict.format_results(task, results))
                history.append({"time": datetime.now().isoformat(timespec="seconds"), "task": task,
                                "title": text, "predictions": {r["model"]: r["label"] for r in results}})
            except Exception as exc:
                print(f"  [error] task {task}: {exc}")
    if history:
        print(f"Classified {len(history) // max(1, len(config.TASKS))} title(s) in this session.")


def main() -> None:
    """Menu loop."""
    while True:
        print_menu()
        choice = ask("Choose an option: ", set(MENU))
        if choice == "0":
            print("Bye!")
            break
        if choice == "11":
            classify_loop()
            continue
        title, module, kwargs, quick_option = MENU[choice]
        quick = ask_yes_no("Quick smoke-test mode (small subset, few epochs)?") if quick_option else False
        print(f"\n>>> {title}")
        for name in ([module] if isinstance(module, str) else module):
            if call_step(name, kwargs, quick=quick) is None:
                break                       # stop a multi-step item at the first failure


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
