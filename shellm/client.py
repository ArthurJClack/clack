"""tiny http client for a running `shellm serve` (standard library only).

    from shellm.client import Client

    c = Client("http://127.0.0.1:8000")
    print(c.complete("The old house on the hill", max_tokens=40))
    for ev in c.stream("Once upon a time", max_tokens=20):
        if ev["type"] == "token":
            print(ev["text"], end="", flush=True)
    c.edit("The captain's horse was named", " Gearheart")
    c.steer(pos=["the sea, the ship"], neg=["the parlour, the tea"], alpha=1.0)
    c.reset()

anything sent through here shows up live in the 3d viewer if one is open.
the server also speaks the openai api at /v1, so `openai.OpenAI(base_url=".../v1",
api_key="x").completions.create(model="shellm-nano", prompt="...")` works too.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator


class ApiError(RuntimeError):
    pass


class Client:
    def __init__(self, url: str = "http://127.0.0.1:8000", timeout: float = 300):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _req(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read())["error"]["message"]
            except Exception:
                msg = str(e)
            raise ApiError(f"{e.code}: {msg}") from None

    def _json(self, method: str, path: str, body: dict | None = None) -> dict:
        with self._req(method, path, body) as r:
            return json.loads(r.read())

    def info(self) -> dict:
        return self._json("GET", "/api/model")

    def state(self) -> dict:
        return self._json("GET", "/api/state")

    def tokenize(self, text: str) -> list[dict]:
        return self._json("POST", "/api/tokenize", {"text": text})["tokens"]

    def generate(self, prompt: str = "", trace: bool = False, **params) -> dict:
        """full result: {"text", "finish_reason", "usage", "tokens": [...]} (tokens carry traces if trace=True)."""
        return self._json("POST", "/api/generate", {"prompt": prompt, "trace": trace, **params})

    def complete(self, prompt: str = "", **params) -> str:
        return self.generate(prompt, **params)["text"]

    def stream(self, prompt: str = "", trace: bool = False, **params) -> Iterator[dict]:
        with self._req("POST", "/api/generate", {"prompt": prompt, "trace": trace, "stream": True, **params}) as r:
            for line in r:
                if line.startswith(b"data: "):
                    yield json.loads(line[6:])

    def steer(self, pos: list[str], neg: list[str], layer: int | None = None, alpha: float = 1.0) -> dict:
        return self._json("POST", "/api/steer", {"pos": pos, "neg": neg, "layer": layer, "alpha": alpha})

    def clear_steering(self) -> dict:
        return self._json("DELETE", "/api/steer")

    def edit(self, prompt: str, target: str, layer: int | None = None) -> dict:
        return self._json("POST", "/api/edit", {"prompt": prompt, "target": target, "layer": layer})

    def reset(self) -> dict:
        return self._json("POST", "/api/reset", {})
