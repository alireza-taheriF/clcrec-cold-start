from __future__ import annotations

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from src.recommend import ColdStartRecommender

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")


def _json_bytes(payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return status, body


def _item_dict(item):
    return {
        "item_id": item.item_id,
        "score": round(item.score, 6),
        "title": item.title,
        "genres": item.genres,
    }


class RecommenderHandler(BaseHTTPRequestHandler):
    recommender: ColdStartRecommender = None
    launch_report: dict = None

    def log_message(self, fmt, *args):
        print("[serve]", self.address_string(), fmt % args)

    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        if not os.path.isfile(path):
            self._send(*_json_bytes({"error": "not found"}, 404))
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype = f"{ctype}; charset=utf-8"
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        rec = self.recommender

        if path in ("/", "/index.html"):
            self._send_file(os.path.join(DASHBOARD_DIR, "index.html"))
            return
        if path.startswith("/static/"):
            name = os.path.basename(path)
            self._send_file(os.path.join(DASHBOARD_DIR, name))
            return
        if path in ("/health", "/v1/health"):
            payload = rec.overview()
            payload["ok"] = True
            payload["catalog"] = payload["n_items"]
            self._send(*_json_bytes(payload))
            return
        if path == "/v1/launch/items":
            self._send(*_json_bytes({"items": rec.list_new_items()}))
            return
        if path == "/v1/launch/compare":
            self._send(*_json_bytes(self.launch_report or {}))
            return
        self._send(*_json_bytes({"error": "not found"}, 404))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send(*_json_bytes({"error": "invalid json"}, 400))
            return

        path = urlparse(self.path).path
        rec = self.recommender
        try:
            if path == "/v1/similar":
                items = rec.similar_items(
                    int(data["item_id"]),
                    k=int(data.get("k", 10)),
                    pool=data.get("pool", "all"),
                )
                payload = {"items": [_item_dict(x) for x in items]}
            elif path == "/v1/recommend":
                if "user_id" in data:
                    items = rec.recommend_user(
                        int(data["user_id"]),
                        k=int(data.get("k", 10)),
                        pool=data.get("pool", "cold"),
                    )
                else:
                    items = rec.recommend_from_likes(
                        data.get("liked_ids", []),
                        k=int(data.get("k", 10)),
                        pool=data.get("pool", "cold"),
                    )
                payload = {"items": [_item_dict(x) for x in items]}
            elif path == "/v1/new_item":
                items = rec.recommend_for_new_item(
                    title=data["title"],
                    genres=data.get("genres", data.get("tags", "(no genres listed)")),
                    year=data.get("year"),
                    k=int(data.get("k", 10)),
                )
                payload = {
                    "query": {"title": data["title"], "genres": data.get("genres")},
                    "neighbors": [_item_dict(x) for x in items],
                }
            elif path == "/v1/launch/plan":
                payload = rec.plan_launch(
                    item_id=int(data["item_id"]) if "item_id" in data else None,
                    title=data.get("title"),
                    tags=data.get("tags") or data.get("genres"),
                    budget=int(data.get("budget", 40)),
                    diversity=float(data.get("diversity", 0.35)),
                )
            else:
                self._send(*_json_bytes({"error": "not found"}, 404))
                return
            self._send(*_json_bytes(payload))
        except (KeyError, ValueError) as exc:
            self._send(*_json_bytes({"error": str(exc)}, 400))


def serve(artifact_dir: str = "artifacts", host: str = "0.0.0.0", port: int = 8080):
    RecommenderHandler.recommender = ColdStartRecommender.load(artifact_dir)
    report_path = os.path.join(artifact_dir, "launch_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            RecommenderHandler.launch_report = json.load(f)
    httpd = ThreadingHTTPServer((host, port), RecommenderHandler)
    print(f"FirstSlot  http://{host}:{port}", flush=True)
    print("  UI   /", flush=True)
    print("  GET  /v1/launch/items", flush=True)
    print("  GET  /v1/launch/compare", flush=True)
    print("  POST /v1/launch/plan     {item_id|title,tags, budget, diversity}", flush=True)
    httpd.serve_forever()
