"""Print shell `export` lines that point dbt at the SAME database as RETAILFC_DSN (so dbt cannot silently test another DB)."""

import os
import shlex
from urllib.parse import urlparse

u = urlparse(os.environ["RETAILFC_DSN"])
for k, v in {
    "DBT_HOST": u.hostname or "localhost",
    "DBT_PORT": str(u.port or 5432),
    "DBT_USER": u.username or "",
    "DBT_PASSWORD": u.password or "",
    "DBT_DBNAME": u.path.lstrip("/"),
}.items():
    print(f"export {k}={shlex.quote(v)}")
