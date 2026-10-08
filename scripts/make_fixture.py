"""Write a SYNTHETIC retail-shaped fixture (for CI/smoke tests only -- never for reported benchmarks).

python scripts/make_fixture.py --out data/fixture_src --days 300
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.synth import write_csvs  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--items", type=int, default=36)
    a = ap.parse_args()
    write_csvs(a.out, a.days, n_items=a.items)
    print(f"SYNTHETIC fixture written to {a.out} (not real data)")
