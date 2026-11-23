"""http api + live 3d viewer for one model. standard library only.

    python -m shellm serve models/shellm-nano            # http://127.0.0.1:8000

  GET  /                       the 3d viewer
  GET  /api/model              architecture, tensor shapes, provenance
  GET  /api/state              current steering + edits
  GET  /api/events             server-sent events: every generation, steer and edit,
                               whoever triggered it (this is what the viewer watches)
  GET  /api/weights?name=&size=&diff=   a small picture of one weight matrix
  POST /api/generate           {prompt, max_tokens, temperature, top_k, top_p, seed, stop, stream, trace}
  POST /api/tokenize           {text}
  POST /api/steer              {pos: [...], neg: [...], layer?, alpha?}     DELETE /api/steer clears it
  POST /api/edit               {prompt, target, layer?}                     rome fact edit, in place
  POST /api/reset              back to the weights on disk
  GET  /v1/models              ┐
  POST /v1/completions         ├ openai-compatible, so existing clients and sdks just work
  POST /v1/chat/completions    ┘ (messages are flattened to text: this is a completion model)

binds to 127.0.0.1 by default and only answers same-origin browser requests.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .api import Session

STATIC = Path(__file__).parent / "viz" / "static"
MAX_BODY = 1 << 20


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def encode_trace(t: dict) -> dict:
    """numpy trace -> compact json (uint8/int8 arrays as base64)."""
    attn = np.clip(np.round(t["attn"] * 255), 0, 255).astype(np.uint8)  # [L, H, S]
    mlp = t["mlp"]
    scale = np.maximum(np.abs(mlp).max(axis=1, keepdims=True), 1e-9)
    return {
        "ctx_offset": t["ctx_offset"],
        "shape": {"layers": int(attn.shape[0]), "heads": int(attn.shape[1]), "ctx": int(attn.shape[2]), "ffn": int(mlp.shape[1])},
        "attn": _b64(attn),
        "mlp": _b64(np.clip(np.round(mlp / scale * 127), -127, 127).astype(np.int8)),
        "mlp_scale": [round(float(s), 4) for s in scale[:, 0]],
        "resid": [[round(float(v), 2) for v in row] for row in t["resid"]],
        "d_attn": [round(v, 3) for v in t["d_attn"]],
        "d_mlp": [round(v, 3) for v in t["d_mlp"]],
        "lens": t["lens"],
        "top": t["top"],
    }


def encode_event(ev: dict) -> dict:
    if "trace" in ev:
        ev = {**ev, "trace": encode_trace(ev["trace"])}
    return ev


class Hub:
    """fan-out of session events to every connected /api/events stream."""

    def __init__(self):
        self.subs: dict[str, queue.Queue] = {}
        self.lock = threading.Lock()

    def active(self) -> bool:
        return bool(self.subs)

    def join(self) -> tuple[str, queue.Queue]:
        sid, q = uuid.uuid4().hex, queue.Queue(maxsize=4096)
        with self.lock:
            self.subs[sid] = q
        return sid, q

    def leave(self, sid: str):
        with self.lock:
            self.subs.pop(sid, None)

    def publish(self, ev: dict):
        if not self.subs:
            return
        data = json.dumps(encode_event(ev))
        with self.lock:
            for q in self.subs.values():
                try:
                    q.put_nowait(data)
                except queue.Full:  # a stalled viewer: drop its oldest event, keep going
                    try:
                        q.get_nowait()
                        q.put_nowait(data)
                    except Exception:
                        pass


class BadRequest(Exception):
    pass


def make_handler(session: Session, hub: Hub, cors: bool, loopback: bool):
    model_id = Path(session.ckpt).name

    def gen_kwargs(body: dict) -> dict:
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        return {
            "max_tokens": max(1, min(int(body.get("max_tokens", 100)), 4096)),
            "temperature": float(body.get("temperature", 0.8)),
            "top_k": body.get("top_k", 50),
            "top_p": body.get("top_p", 0.95),
            "repetition_penalty": float(body.get("repetition_penalty", 1.1)),
            "seed": body.get("seed"),
            "stop": stop,
        }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "shellm"

        def log_message(self, fmt, *args):
            if not self.path.startswith("/static"):
                print(f"[serve] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

        # ---------------------------------------------------------- plumbing
        def _guard(self) -> bool:
            if loopback:  # dns-rebinding guard: a loopback server only answers to loopback names
                host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
                if host not in ("127.0.0.1", "localhost", "::1"):
                    self._json({"error": {"message": "bad host header", "type": "forbidden"}}, 403)
                    return False
            return True

        def _headers(self, code: int, ctype: str, length: int | None = None, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if length is not None:
                self.send_header("Content-Length", str(length))
            if cors:
                self.send_header("Access-Control-Allow-Origin", "*")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def _json(self, obj, code: int = 200):
            data = json.dumps(obj).encode()
            self._headers(code, "application/json", len(data))
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise BadRequest("body too large")
            if n and "json" not in (self.headers.get("Content-Type") or ""):
                raise BadRequest("send Content-Type: application/json")
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError as e:
                raise BadRequest(f"invalid json: {e}")
            if not isinstance(body, dict):
                raise BadRequest("body must be a json object")
            return body

        def _sse_start(self):
            self._headers(200, "text/event-stream", extra={"Cache-Control": "no-cache", "Connection": "close", "X-Accel-Buffering": "no"})
            self.close_connection = True

        def _sse(self, data: str):
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()

        def _route(self, method: str):
            if not self._guard():
                return
            url = urlparse(self.path)
            try:
                fn = ROUTES.get((method, url.path))
                if fn is None and method == "GET" and (url.path == "/" or url.path.startswith("/static/")):
                    return self._static(url.path)
                if fn is None:
                    return self._json({"error": {"message": f"no route {method} {url.path}", "type": "not_found"}}, 404)
                fn(self, url)
            except BadRequest as e:
                self._json({"error": {"message": str(e), "type": "invalid_request_error"}}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (KeyError, ValueError, TypeError) as e:
                self._json({"error": {"message": f"{type(e).__name__}: {e}", "type": "invalid_request_error"}}, 400)
            except Exception as e:  # noqa: BLE001 - last-resort 500 so the server stays up
                self._json({"error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}}, 500)

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def do_DELETE(self):
            self._route("DELETE")

        def do_OPTIONS(self):
            if cors:
                self._headers(204, "text/plain", 0, {"Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                                                     "Access-Control-Allow-Headers": "Content-Type, Authorization"})
            else:
                self._json({"error": {"message": "cross-origin requests are off (start with --cors)", "type": "forbidden"}}, 403)

        def _static(self, path: str):
            rel = "index.html" if path == "/" else path.removeprefix("/static/")
            f = (STATIC / rel).resolve()
            if STATIC.resolve() not in f.parents or not f.is_file():
                return self._json({"error": {"message": "not found", "type": "not_found"}}, 404)
            data = f.read_bytes()
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            if f.suffix == ".js":
                ctype = "text/javascript"
            self._headers(200, ctype, len(data), {"Cache-Control": "no-cache"})
            self.wfile.write(data)

        # ---------------------------------------------------------- native api
        def api_model(self, url):
            self._json(session.info())

        def api_state(self, url):
            self._json(session.state())

        def api_health(self, url):
            self._json({"ok": True, "model": model_id, "viewers": len(hub.subs)})

        def api_weights(self, url):
            q = parse_qs(url.query)
            name = q.get("name", [""])[0]
            if name not in session.W.t:
                raise BadRequest(f"unknown tensor {name!r}")
            size = max(4, min(int(q.get("size", ["32"])[0]), 128))
            img = session.weight_thumb(name, size, diff=q.get("diff", ["0"])[0] in ("1", "true"))
            mx = float(img.max()) or 1.0
            self._json({"name": name, "rows": int(img.shape[0]), "cols": int(img.shape[1]), "max": mx,
                        "data": _b64(np.clip(np.round(img / mx * 255), 0, 255).astype(np.uint8))})

        def api_events(self, url):
            sid, q = hub.join()
            try:
                self._sse_start()
                self._sse(json.dumps({"type": "hello", "model": session.info(), "state": session.state()}))
                while True:
                    try:
                        self._sse(q.get(timeout=15))
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                hub.leave(sid)

        def api_tokenize(self, url):
            body = self._body()
            toks = session.tokenize(str(body.get("text", "")), bos=bool(body.get("bos", True)))
            self._json({"tokens": toks, "count": len(toks)})

        def api_generate(self, url):
            body = self._body()
            kw = gen_kwargs(body)
            trace = bool(body.get("trace", False))
            gen = session.generate(str(body.get("prompt", "")), trace=trace, source=str(body.get("source", "http")), **kw)
            try:
                if body.get("stream"):
                    self._sse_start()
                    for ev in gen:
                        self._sse(json.dumps(encode_event(ev)))
                else:
                    evs = [encode_event(ev) for ev in gen]
                    out = {**evs[-1], "tokens": [e for e in evs if e["type"] == "token"]}
                    out.pop("type")
                    self._json(out)
            finally:
                gen.close()

        def api_steer(self, url):
            body = self._body()
            pos, neg = body.get("pos"), body.get("neg")
            if not (isinstance(pos, list) and isinstance(neg, list) and pos and neg):
                raise BadRequest("steer needs non-empty lists: pos and neg")
            self._json(session.steer([str(p) for p in pos], [str(n) for n in neg], body.get("layer"), float(body.get("alpha", 1.0))))

        def api_steer_clear(self, url):
            session.clear_steering()
            self._json({"ok": True})

        def api_edit(self, url):
            body = self._body()
            if not body.get("prompt") or not body.get("target"):
                raise BadRequest("edit needs prompt and target")
            self._json(session.edit(str(body["prompt"]), str(body["target"]), body.get("layer")))

        def api_reset(self, url):
            session.reset()
            self._json({"ok": True, **session.state()})

        # ---------------------------------------------------------- openai-compatible
        def v1_models(self, url):
            self._json({"object": "list", "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "you"}]})

        def _openai(self, prompt: str, body: dict, chat: bool):
            kw = gen_kwargs(body)
            if "max_completion_tokens" in body:
                kw["max_tokens"] = max(1, min(int(body["max_completion_tokens"]), 4096))
            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
            base = {"id": rid, "object": "chat.completion" if chat else "text_completion", "created": int(time.time()), "model": model_id}

            def choice(text, finish, delta=False):
                if chat:
                    key = "delta" if delta else "message"
                    msg = {"content": text} if delta else {"role": "assistant", "content": text}
                    return [{"index": 0, key: msg, "finish_reason": finish}]
                return [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]

            gen = session.generate(prompt, source="openai-api", **kw)
            try:
                if body.get("stream"):
                    self._sse_start()
                    chunk = {**base, "object": "chat.completion.chunk"} if chat else base
                    for ev in gen:
                        if ev["type"] == "token":
                            self._sse(json.dumps({**chunk, "choices": choice(ev["text"], None, delta=True)}))
                        elif ev["type"] == "done":
                            self._sse(json.dumps({**chunk, "choices": choice("", ev["finish_reason"], delta=True)}))
                    self._sse("[DONE]")
                else:
                    done = [ev for ev in gen][-1]
                    u = done["usage"]
                    self._json({**base, "choices": choice(done["text"], done["finish_reason"]),
                                "usage": {"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
                                          "total_tokens": u["prompt_tokens"] + u["completion_tokens"]}})
            finally:
                gen.close()

        def v1_completions(self, url):
            body = self._body()
            prompt = body.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            self._openai(str(prompt), body, chat=False)

        def v1_chat(self, url):
            body = self._body()
            parts = []
            for m in body.get("messages") or []:
                c = m.get("content", "")
                if isinstance(c, list):
                    c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
                parts.append(str(c))
            self._openai("\n\n".join(parts), body, chat=True)

    ROUTES = {
        ("GET", "/api/model"): Handler.api_model,
        ("GET", "/api/state"): Handler.api_state,
        ("GET", "/api/events"): Handler.api_events,
        ("GET", "/api/weights"): Handler.api_weights,
        ("GET", "/health"): Handler.api_health,
        ("POST", "/api/generate"): Handler.api_generate,
        ("POST", "/api/tokenize"): Handler.api_tokenize,
        ("POST", "/api/steer"): Handler.api_steer,
        ("DELETE", "/api/steer"): Handler.api_steer_clear,
        ("POST", "/api/edit"): Handler.api_edit,
        ("POST", "/api/reset"): Handler.api_reset,
        ("GET", "/v1/models"): Handler.v1_models,
        ("POST", "/v1/completions"): Handler.v1_completions,
        ("POST", "/v1/chat/completions"): Handler.v1_chat,
    }
    return Handler


def make_server(ckpt, host: str = "127.0.0.1", port: int = 8000, device: str = "auto", data: str | None = None,
                cors: bool = False) -> tuple[ThreadingHTTPServer, Session]:
    session = Session(ckpt, device=device, data=data)
    hub = Hub()

    def publish(ev):
        hub.publish(ev)

    publish.active = hub.active  # the session only pays for tracing while a viewer is connected
    session.subscribe(publish)
    loopback = host in ("127.0.0.1", "localhost", "::1")
    httpd = ThreadingHTTPServer((host, port), make_handler(session, hub, cors, loopback))
    httpd.daemon_threads = True
    return httpd, session


def serve(ckpt, host: str = "127.0.0.1", port: int = 8000, device: str = "auto", data: str | None = None, cors: bool = False):
    httpd, session = make_server(ckpt, host, port, device, data, cors)
    c = session.cfg
    print(f"[serve] {Path(session.ckpt).name}: {c.n_layers} layers x {c.n_heads} heads, {session.info()['n_params']/1e6:.1f}m params on {session.device}")
    print(f"[serve] viewer  http://{host}:{httpd.server_address[1]}/")
    print(f"[serve] api     http://{host}:{httpd.server_address[1]}/v1/completions  (openai-compatible)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye")
    finally:
        httpd.server_close()
