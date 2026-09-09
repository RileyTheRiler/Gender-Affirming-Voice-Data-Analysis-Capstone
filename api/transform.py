from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from api.index import MAX_WAV_BASE64_CHARS, TransformRequest, transform


# Vercel's /api directory is file-routed. Keeping a concrete
# api/transform.py entrypoint guarantees that POST /api/transform reaches
# Python instead of falling through to Vercel's plain-text NOT_FOUND page.
MAX_BODY_BYTES = MAX_WAV_BASE64_CHARS + 32_768


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send_json(200, {"ok": True, "route": "/api/transform", "method": "POST"})

    def do_POST(self) -> None:
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"detail": "Invalid Content-Length header."})
                return

            if length <= 0:
                self._send_json(400, {"detail": "Request body is required."})
                return
            if length > MAX_BODY_BYTES:
                self._send_json(413, {"detail": "Audio request is too large. Record 15 seconds or less."})
                return

            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8"))
            req = TransformRequest.model_validate(payload)
            result = transform(req)
            self._send_json(200, result)
        except json.JSONDecodeError:
            self._send_json(400, {"detail": "Request body must be valid JSON."})
        except UnicodeDecodeError:
            self._send_json(400, {"detail": "Request body must be UTF-8 JSON."})
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            message = first.get("msg") or "Invalid transform request."
            self._send_json(422, {"detail": message})
        except HTTPException as exc:
            self._send_json(int(exc.status_code), {"detail": str(exc.detail)})
        except Exception:
            self._send_json(500, {"detail": "Transformation failed unexpectedly."})
