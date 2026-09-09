from http.server import BaseHTTPRequestHandler

from api.index import MAX_WAV_BASE64_CHARS
from api.transform import MAX_BODY_BYTES, handler


def test_vercel_transform_entrypoint_exports_http_handler():
    assert issubclass(handler, BaseHTTPRequestHandler)


def test_vercel_entrypoint_caps_body_before_json_parse():
    assert MAX_BODY_BYTES > MAX_WAV_BASE64_CHARS
    assert MAX_BODY_BYTES < MAX_WAV_BASE64_CHARS + 100_000
