"""Check native thread limits inside real prediction workers."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_prediction_workers_use_one_native_thread(tmp_path):
    # OpenMP reads this environment when a fresh Python process loads the model.
    script = textwrap.dedent(
        """
        import json
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        from fastapi.testclient import TestClient
        from threadpoolctl import threadpool_info

        from bike_demand.api import EXAMPLE_REQUEST, create_app

        app = create_app(
            Path("artifacts/run_01/model.joblib"), log_path=Path(sys.argv[1])
        )
        observations = []
        with TestClient(app) as client:
            original_predict = app.state.estimator.predict

            def checked_predict(features):
                observations.append([
                    {"api": pool["user_api"], "threads": pool["num_threads"]}
                    for pool in threadpool_info()
                ])
                return original_predict(features)

            app.state.estimator.predict = checked_predict

            def send(_):
                response = client.post("/predict", json=EXAMPLE_REQUEST)
                assert response.status_code == 200

            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(send, range(4)))

        print(json.dumps(observations))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "requests.jsonl")],
        cwd=PROJECT_DIR,
        env=os.environ | {"OMP_NUM_THREADS": "4"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    observations = json.loads(result.stdout)
    assert len(observations) == 4
    for pools in observations:
        assert any(pool["api"] == "openmp" for pool in pools)
        assert all(pool["threads"] == 1 for pool in pools), pools
