from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from src.recommend import ColdStartRecommender


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

    def log_message(self, fmt, *args):
        print("[serve]", self.address_string(), fmt % args)

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/health", "/v1/health"):
            status, body = _json_bytes({"ok": True, "catalog": len(self.recommender.item_emb)})
        else:
            status, body = _json_bytes({"error": "not found"}, 404)
        self._send(status, body)

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
                    genres=data.get("genres", "(no genres listed)"),
                    year=data.get("year"),
                    k=int(data.get("k", 10)),
                )
                payload = {
                    "query": {"title": data["title"], "genres": data.get("genres")},
                    "neighbors": [_item_dict(x) for x in items],
                }
            else:
                self._send(*_json_bytes({"error": "not found"}, 404))
                return
            self._send(*_json_bytes(payload))
        except (KeyError, ValueError) as exc:
            self._send(*_json_bytes({"error": str(exc)}, 400))


def serve(artifact_dir: str = "artifacts", host: str = "0.0.0.0", port: int = 8080):
    RecommenderHandler.recommender = ColdStartRecommender.load(artifact_dir)
    httpd = ThreadingHTTPServer((host, port), RecommenderHandler)
    print(f"CLCRec Lab API  http://{host}:{port}")
    print("  GET  /v1/health")
    print("  POST /v1/similar     {item_id, k, pool}")
    print("  POST /v1/recommend   {user_id|liked_ids, k, pool}")
    print("  POST /v1/new_item    {title, genres, year, k}")
    httpd.serve_forever()
