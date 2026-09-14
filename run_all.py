"""
QF600 -- run the whole pipeline end to end, in order.

Each step is independently runnable (python src/stepN_*.py); this just chains them
so the full result set can be regenerated from raw data in one command.

    python run_all.py             # full run (network fetches are cached)
    python run_all.py --from 4    # resume from a given step

Steps 1-3 hit the network and cache to data/raw and data/clean; re-runs are cheap.
Steps 4-9 are pure computation over those caches.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "bin" / "python")

STEPS = [
    (1, "step1_load.py", "Load, filter, clean disclosures"),
    (2, "step1_diagnostic.py", "Trade-count diagnostic (fixes the split)"),
    (3, "step2_data.py", "Prices and factors"),
    (4, "step3_holding.py", "Holding-period panel (FOUNDATIONAL)"),
    (5, "step4_formation.py", "Track A selection"),
    (6, "step5_committee.py", "Track B selection (committee)"),
    (7, "step6_portfolio.py", "Portfolio return series"),
    (8, "step7_regression.py", "CAPM / 4-factor regressions"),
    (9, "step8_robustness.py", "Robustness -- all variants"),
]


def main():
    start = 1
    if "--from" in sys.argv:
        start = int(sys.argv[sys.argv.index("--from") + 1])

    for n, script, desc in STEPS:
        if n < start:
            print(f"[{n}/9] SKIP  {desc}")
            continue
        print(f"\n{'='*78}\n[{n}/9] {desc}\n  -> src/{script}\n{'='*78}")
        r = subprocess.run([PY, str(ROOT / "src" / script)])
        if r.returncode != 0:
            print(f"\nStep {n} ({script}) FAILED with code {r.returncode}. Stopping.")
            sys.exit(r.returncode)

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE. Reports in data/reports/:")
    for p in sorted((ROOT / "data" / "reports").glob("step*.txt")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
