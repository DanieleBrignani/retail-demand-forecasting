"""Builds notebooks/retail_walkthrough.ipynb. The notebook only calls shared retailfc functions (no separate pipeline)."""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = lambda s: nbf.v4.new_code_cell(s)
M = lambda s: nbf.v4.new_markdown_cell(s)
nb.cells = [
    M(
        "# Retail demand forecasting — walkthrough\n\nEvery cell calls the same library functions the CLI, Prefect flows and tests use "
        "(`retailfc.panel`, `retailfc.features`, `retailfc.eda`, `retailfc.experiments`, `retailfc.report`). Set `RETAILFC_DATA_DIR` / `RETAILFC_ARTIFACTS_DIR` "
        "and optionally `RETAILFC_DATASET_VERSION` before starting Jupyter. **Results appear only if the experiments were run; otherwise sections say PENDING.**"
    ),
    C(
        "import os, json, pandas as pd\nfrom pathlib import Path\nfrom IPython.display import Markdown, Image, display\n"
        "from retailfc import config as C, experiments as E, protocol as P, report\nfrom retailfc.ingestion import raw\nfrom retailfc.panel import load_panel_from_raw\n"
        "cfg = C.load_config(os.environ.get('RETAILFC_CONFIG'))\nversion = os.environ.get('RETAILFC_DATASET_VERSION') or raw.latest_version()\n"
        "manifest = raw.read_manifest(version)\nprint(version, manifest['observed_sales'])"
    ),
    M(
        "## Dataset and information boundary\nThe manifest records source, checksums, schema version and observed date range."
    ),
    C("pd.DataFrame(manifest['files']).T"),
    C(
        "panel = load_panel_from_raw(version, max_series=int(os.environ.get('RETAILFC_NB_MAX_SERIES', 0)) or None)\n"
        "origins = P.make_origins(panel.n_days, cfg)\nprint('series', panel.n_series, 'days', panel.n_days, origins)"
    ),
    M(
        "## Feature dictionary and leakage contract\nGenerated from `retailfc.features.spec` (single source of truth)."
    ),
    C(
        "from retailfc.features.spec import SPECS\npd.DataFrame([s.__dict__ for s in SPECS])[['name','group','definition','time_ref','availability','missing']]"
    ),
    M("## EDA (training period only)"),
    C(
        "out = Path(os.environ.get('RETAILFC_ARTIFACTS_DIR', 'artifacts')) / 'eda_nb'\nres = E_eda = __import__('retailfc.eda', fromlist=['x']).run_eda(panel, cfg, out)\n"
        "for f in res['findings']: print('-', f)\nfor name in res['charts'].values(): display(Image(str(out / name)))"
    ),
    M("## Experiment results (from stored outputs)"),
    C(
        "exp_id = os.environ.get('RETAILFC_EXPERIMENT_ID') or E.make_experiment_id(cfg, version)\n"
        "r = report.build_report(exp_id)\nprint(r['status'])\ndisplay(Markdown((Path(r['out']) / 'results.md').read_text(encoding='utf-8')))"
    ),
]
nbf.write(nb, Path(__file__).parent / "retail_walkthrough.ipynb")
print("ok")
