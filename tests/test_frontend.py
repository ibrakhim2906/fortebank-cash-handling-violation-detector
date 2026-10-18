"""Presentation integration: assets, aggregate privacy, and unchanged jury API."""
import asyncio
import json
from pathlib import Path

from service.app import create_app
from test_video_api import Client, KEY


def test_frontend_assets_do_not_expose_source_artifacts(tmp_path):
    async def scenario():
        app = create_app(api_key=KEY, jobs_dir=tmp_path)
        async with Client(app) as client:
            # FileResponse uses the modern ASGI send path. The shared test
            # driver's immediate disconnect otherwise cancels streamed bodies.
            async def modern_asgi(scope, receive, send):
                scope = {**scope, "asgi": {**scope["asgi"], "spec_version": "2.4"}}
                await app(scope, receive, send)
            client.app = modern_asgi
            status, html = await client.request("GET", "/")
            assert status == 200 and 'lang="en"' in html
            assert 'data-page="review"' in html and 'id="upload"' in html
            assert 'class="breadcrumb"' not in html and 'class="product-name"' not in html
            assert 'id="sidebar-toggle"' in html and 'id="workspace-sidebar"' in html
            # The current review layout has an event list; its DOM id does not
            # expose source artifacts. Keep the obsolete KPI surface absent.
            assert 'id="kpis"' not in html and 'id="events"' in html
            assert html.count('class="nav-item') == 2
            assert 'href="/results"' not in html
            status, evaluation = await client.request("GET", "/evaluation")
            assert status == 200 and "ACCURACY TARGET · NOT YET ACHIEVED" in evaluation
            assert 'id="upload"' not in evaluation
            status, results = await client.request("GET", "/results")
            assert status == 200 and 'id="events"' in results
            # Results shares the review workspace, including a new-upload form.
            assert 'id="kpis"' not in results and 'id="upload"' in results
            assert results == html
            assert (await client.request("GET", "/review"))[0] == 200
            for path in ["/styles/tokens.css", "/static/workspace.css", "/static/theme.js", "/static/shell.js", "/static/workspace.js", "/static/review.js", "/static/results.js", "/static/evaluation.js", "/static/evaluation.json"]:
                assert (await client.request("GET", path))[0] == 200
            for path in ["/data/processed/mission-32b-proposal-dev30-report.json", "/data/raw/", "/static/../app.py"]:
                assert (await client.request("GET", path))[0] == 404
            assert (await client.status("unknown", key=""))[0] == 401
    asyncio.run(scenario())


def test_bundled_snapshot_is_measured_aggregates_only():
    from scripts.build_frontend_snapshot import build_snapshot
    snapshot = json.loads((Path(__file__).resolve().parents[1] / "service/static/evaluation.json").read_text())
    assert snapshot == build_snapshot()
    assert snapshot["target_status"] == "aspirational_not_achieved"
    assert snapshot["independent_test"] is False
    for row in snapshot["recipes"]:
        assert row["accuracy"] == row["correct"] / row["n_videos"]
        assert not {"per_video", "predictions", "aliases", "group_id", "evidence_frames", "image_base64"}.intersection(row)
    assert snapshot["checkpoint"]["worst_sec"] == 72.98429473300348 and snapshot["checkpoint"]["failed_jobs"] == 0
