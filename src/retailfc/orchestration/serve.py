"""Register scheduled Prefect deployments (run inside the `prefect-worker` container or locally).

Schedules are intentionally modest: the evaluation flow runs daily but only *decides*; it retrains only when the policy
in config (`retraining`) is met.
"""

from __future__ import annotations

from datetime import date

from prefect import serve

from .flows import actuals_monitoring_flow, retraining_evaluation_flow


def main() -> None:
    serve(
        retraining_evaluation_flow.to_deployment(
            name="daily-retraining-evaluation",
            cron="0 3 * * *",
            parameters={"as_of": date.today().isoformat()},
        ),
        actuals_monitoring_flow.to_deployment(name="manual-actuals-monitoring"),
    )


if __name__ == "__main__":
    main()
