"""Shared judge-API transport (OpenAI-compatible chat/completions).

A single, dependency-light client around the gateway used by *every*
judge-backed component in the pipeline:

  * :mod:`emo.measure.llm_judge`    — 0-100 emotion-intensity ratings,
  * :mod:`emo.measure.encoder_judge`— forced distribution over the 7 ISEAR
    emotions (the API replacement for the RoBERTa encoder),
  * :mod:`emo.data.build_pairs`     — contrastive pos/neg pair generation (the
    API replacement for the local Qwen2.5-3B generator).

Centralising the HTTP transport keeps retry/backoff, ``.env`` loading and JSON
extraction in one place, so the rest of the code only deals with prompts and
parsed results. Endpoint, token and model come from ``.env`` (see
``.env.example``); network failures return ``None`` rather than raising, so a
caller can degrade gracefully (skip a text, fall back, etc.).

Timeouts are split into a short *connect* timeout (fail fast when the gateway
is unreachable — the symptom that used to hang the whole sweep for minutes) and
a longer *read* timeout (generation can legitimately take a while). A
:meth:`JudgeClient.ping` preflight is provided so callers can check reachability
*once* and abort early with a clear message instead of timing out on every
single request. :meth:`JudgeClient.complete_many` fans a batch of requests out
across a small thread pool (stdlib only — no extra dependency) so the many
independent judge calls in Stage0/Stage3 run in parallel.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests


# Configuration source of truth is the process environment (os.environ), i.e.
# plain `export JUDGE_API_URL=... JUDGE_API_TOKEN=... JUDGE_MODEL=...` before
# launching. A local emo_exp/.env is supported only as an optional convenience
# for dev machines and is loaded ONLY when the file actually exists *and*
# python-dotenv is installed; it never overrides variables already exported
# (override=False), and its absence is completely silent. Set
# JUDGE_DISABLE_DOTENV=1 to skip the .env lookup entirely.
def _maybe_load_dotenv() -> None:
    if os.environ.get("JUDGE_DISABLE_DOTENV", "").strip() in ("1", "true", "True"):
        return
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
    if not os.path.exists(env_path):
        return  # no .env: rely purely on exported environment variables
    try:
        from dotenv import load_dotenv

        # override=False -> exported env vars win over the file.
        load_dotenv(env_path, override=False)
    except Exception:
        pass


_maybe_load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


class JudgeClient:
    """Thin OpenAI-compatible chat/completions client over the gateway.

    Parameters mirror the ``.env`` keys; anything left ``None`` falls back to
    the environment. ``model`` may be overridden per-call (the generator and
    the scorers may legitimately use different judge models).

    Parameters
    ----------
    timeout:
        Read timeout in seconds (how long one generation may take). The
        *connect* timeout is separate and short (``connect_timeout``) so an
        unreachable gateway fails in seconds rather than minutes.
    connect_timeout:
        TCP-connect timeout in seconds. Defaults to ``JUDGE_CONNECT_TIMEOUT``
        env or 10s. This is the knob that stops the "connect timed out" hangs.
    workers:
        Default thread-pool size for :meth:`complete_many`. Defaults to
        ``JUDGE_WORKERS`` env or 8.
    """

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        connect_timeout: float | None = None,
        workers: int | None = None,
        proxy: str | None = None,
        verify: bool | str | None = None,
    ):
        self.url = url or os.environ.get("JUDGE_API_URL")
        self.token = token or os.environ.get("JUDGE_API_TOKEN")
        self.model = model or os.environ.get("JUDGE_MODEL", "plus-support-qwen-lite")
        self.timeout = timeout
        self.max_retries = max_retries
        self.connect_timeout = (
            connect_timeout
            if connect_timeout is not None
            else float(_env_int("JUDGE_CONNECT_TIMEOUT", 10))
        )
        self.workers = int(workers or _env_int("JUDGE_WORKERS", 8))
        if not self.url:
            raise ValueError(
                "JUDGE_API_URL is not set. Export it before launching, e.g.:\n"
                "  export JUDGE_API_URL=https://<host>/v1/chat/completions\n"
                "  export JUDGE_API_TOKEN=<token>\n"
                "  export JUDGE_MODEL=<model>\n"
                "No .env file is required — environment variables are the "
                "source of truth."
            )
        # A pooled session is reused across requests/threads (requests.Session
        # is thread-safe for plain request calls and keeps keep-alive
        # connections warm, which matters when fanning out many calls).
        self._session = requests.Session()

        # Proxy handling. On many internal pods an outbound HTTPS proxy is
        # required to reach the gateway: `curl` may already use one (via
        # ~/.curlrc or shell config) while Python's requests does not see it,
        # which is the usual cause of "curl works but requests times out on
        # connect". Resolve a proxy explicitly so both behave the same:
        #   1. explicit `proxy` arg,
        #   2. JUDGE_API_PROXY env (judge-specific, avoids surprising other libs),
        #   3. standard HTTPS_PROXY / https_proxy env.
        # Set JUDGE_NO_PROXY=1 to force a direct connection (ignore all of the
        # above and any trust_env proxies), e.g. when the gateway is reachable
        # directly and an inherited proxy is the thing that's broken.
        self.no_proxy = os.environ.get("JUDGE_NO_PROXY", "").strip() in (
            "1",
            "true",
            "True",
        )
        self.proxy = (
            proxy
            or os.environ.get("JUDGE_API_PROXY")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
        )
        if self.no_proxy:
            # Ignore environment proxies entirely and connect directly.
            self._session.trust_env = False
            self._session.proxies = {}
            self.proxy = None
        elif self.proxy:
            # Pin the chosen proxy for both schemes so it's deterministic.
            self._session.proxies = {"http": self.proxy, "https": self.proxy}

        # TLS certificate verification. Internal gateways often present a
        # corporate / self-signed CA that isn't in the default trust store, so
        # a verified request can fail (sometimes surfacing as a connection-style
        # error). Allow turning verification off — mirroring the `verify=False`
        # used elsewhere in this repo's clients — via JUDGE_VERIFY_SSL=0 (or the
        # judge.verify_ssl config key passed through from_config). Default stays
        # True (secure). A custom CA bundle path is also accepted.
        verify_env = os.environ.get("JUDGE_VERIFY_SSL")
        if verify is not None:
            self.verify = verify
        elif verify_env is not None:
            v = verify_env.strip()
            if v in ("0", "false", "False", "no"):
                self.verify = False
            elif v in ("1", "true", "True", "yes"):
                self.verify = True
            else:
                self.verify = v  # treat as a CA bundle path
        else:
            self.verify = True
        if self.verify is False:
            # Silence the noisy per-request InsecureRequestWarning when the user
            # has explicitly opted out of verification.
            try:
                import urllib3

                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass

        self._reachable: bool | None = None
        self._reach_lock = threading.Lock()

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(cls, cfg, model: str | None = None) -> "JudgeClient":
        """Build a client from a :class:`emo.config.Config` ``judge`` block.

        Pulls worker count and the split connect/read timeouts from
        ``cfg.judge`` (with env/defaults as the fallback) so every judge-backed
        component in a run shares the same transport tuning.
        """
        j = getattr(cfg, "judge", None) or {}
        return cls(
            model=model,
            timeout=int(j.get("read_timeout", 120)),
            connect_timeout=float(
                j.get("connect_timeout", _env_int("JUDGE_CONNECT_TIMEOUT", 10))
            ),
            workers=int(j.get("workers", _env_int("JUDGE_WORKERS", 8))),
            # None -> fall back to JUDGE_VERIFY_SSL env / secure default.
            verify=j.get("verify_ssl", None),
        )

    # -- transport ----------------------------------------------------------

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    @property
    def _timeout_pair(self) -> tuple[float, float]:
        """(connect, read) timeout tuple passed to requests."""
        return (self.connect_timeout, float(self.timeout))

    def ping(self, verbose: bool = True) -> bool:
        """One-shot reachability/auth preflight.

        Sends a tiny completion and caches the result. Returns ``True`` when the
        gateway answered (any HTTP status that isn't a connection failure counts
        as "reachable"; a 4xx still means the host is up). On a connection
        failure it returns ``False`` and prints a clear, actionable message so
        the caller can abort the sweep instead of letting every request time
        out. Subsequent calls return the cached verdict.
        """
        with self._reach_lock:
            if self._reachable is not None:
                return self._reachable
            host = self.url
            try:
                r = self._session.post(
                    self.url,
                    headers=self._headers(),
                    data=json.dumps(
                        {
                            "model": self.model,
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1,
                            "stream": False,
                            "tags": ["batch"],
                            "separate_reasoning": True,
                            "chat_template_kwargs": {"enable_thinking": False},
                        }
                    ),
                    timeout=(self.connect_timeout, min(30.0, float(self.timeout))),
                    verify=self.verify,
                )
                # Host answered (even an error status means it's reachable).
                self._reachable = True
                if verbose:
                    px = self.proxy or ("direct" if self.no_proxy else "none")
                    print(
                        f"[judge] preflight OK: {host} "
                        f"(model={self.model}, http={r.status_code}, proxy={px}, "
                        f"verify={self.verify})",
                        flush=True,
                    )
            except requests.exceptions.RequestException as e:
                self._reachable = False
                if verbose:
                    self._print_preflight_failure(host, e)
            return self._reachable

    def _print_preflight_failure(self, host: str, e: Exception) -> None:
        """Print an actionable diagnosis distinguishing proxy/TCP/DNS issues."""
        import socket
        from urllib.parse import urlparse

        tok = "set" if self.token else "MISSING"
        parsed = urlparse(host)
        dest_host = parsed.hostname or "?"
        dest_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        proxy = self.proxy if not self.no_proxy else "(disabled via JUDGE_NO_PROXY)"

        # Raw TCP probe straight to host:port (no proxy, no TLS) to tell apart
        # "the host itself is blocked" from "the proxy/requests config is wrong".
        tcp_note = ""
        try:
            with socket.create_connection(
                (dest_host, dest_port), timeout=self.connect_timeout
            ):
                tcp_note = (
                    f"raw TCP connect to {dest_host}:{dest_port} SUCCEEDED — the "
                    f"host is reachable directly, so the failure is in the HTTP "
                    f"layer (likely a proxy). Try `export JUDGE_NO_PROXY=1` to "
                    f"force a direct connection, or set JUDGE_API_PROXY to the "
                    f"correct proxy."
                )
        except OSError as se:
            tcp_note = (
                f"raw TCP connect to {dest_host}:{dest_port} ALSO FAILED ({se}) — "
                f"the host:port is blocked from this pod (firewall/VPN), not a "
                f"code or proxy issue. curl may be succeeding via a proxy; set "
                f"JUDGE_API_PROXY to that same proxy."
            )

        # SSL/TLS errors look different from connect timeouts: hint at the
        # verify toggle when the failure mentions certificates/SSL.
        tls_note = ""
        if "SSL" in str(e) or "certificate" in str(e).lower():
            tls_note = (
                "\n        TLS note: this looks like a certificate verification "
                "failure. If the gateway uses an internal CA, set "
                "`export JUDGE_VERIFY_SSL=0` (or judge.verify_ssl=false), or "
                "point JUDGE_VERIFY_SSL at a CA bundle path."
            )

        print(
            f"[judge] PREFLIGHT FAILED: cannot reach {host}\n"
            f"        reason: {e}\n"
            f"        destination: host={dest_host} port={dest_port} "
            f"(https forces 443; the code never uses port 80)\n"
            f"        proxy in use: {proxy}  |  verify_ssl: {self.verify}\n"
            f"        diagnostic: {tcp_note}{tls_note}\n"
            f"        resolved config: JUDGE_API_URL={host} "
            f"JUDGE_API_TOKEN={tok} JUDGE_MODEL={self.model}\n"
            f"        Aborting judge calls early so the sweep doesn't hang on "
            f"every request. Set judge.preflight=false to skip this check.",
            flush=True,
        )

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 1000,
        temperature: float | None = None,
        model: str | None = None,
        label: str | None = None,
    ) -> str | None:
        """Run one chat completion; return the assistant content or ``None``.

        Retries on read-timeout / transient errors with linear backoff. A
        *connect* timeout (gateway unreachable) is not worth retrying three
        times — it fails fast after one short attempt. Every failure log line
        carries ``label`` (e.g. the emotion/text being processed) so a hang can
        be traced to the exact item instead of an anonymous "request timed out".
        """
        tag = f"[judge:{label}]" if label else "[judge]"
        payload: dict = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            # Routing tag expected by the gateway (matches the known-good
            # test_connection.py request that reaches the batch backend).
            "tags": ["batch"],
            # Qwen-specific: keep reasoning out of the visible content.
            "separate_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        for attempt in range(self.max_retries):
            try:
                r = self._session.post(
                    self.url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self._timeout_pair,
                    verify=self.verify,
                )
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except requests.exceptions.ConnectTimeout as e:
                # Gateway unreachable: do not waste three long retries on it.
                print(
                    f"{tag} connect timed out ({self.connect_timeout:.0f}s) — "
                    f"gateway unreachable, not retrying: {e}",
                    flush=True,
                )
                self._reachable = False
                return None
            except requests.exceptions.ConnectionError as e:
                print(f"{tag} connection error, not retrying: {e}", flush=True)
                self._reachable = False
                return None
            except requests.exceptions.ReadTimeout as e:
                if attempt == self.max_retries - 1:
                    print(f"{tag} read timed out (final): {e}", flush=True)
                    return None
                print(
                    f"{tag} read timed out (attempt {attempt + 1}/"
                    f"{self.max_retries}), retrying...",
                    flush=True,
                )
                time.sleep(1.5 * (attempt + 1))
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"{tag} request failed (final): {e}", flush=True)
                    return None
                print(
                    f"{tag} request failed (attempt {attempt + 1}/"
                    f"{self.max_retries}), retrying: {e}",
                    flush=True,
                )
                time.sleep(1.5 * (attempt + 1))
        return None

    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float | None = None,
        model: str | None = None,
        label: str | None = None,
    ) -> str | None:
        """Convenience wrapper: single user prompt (+ optional system)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            label=label,
        )

    # -- parallel fan-out ---------------------------------------------------

    def complete_text_many(
        self,
        prompts: list[str],
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float | None = None,
        model: str | None = None,
        workers: int | None = None,
        label_prefix: str = "",
        progress_every: int = 10,
    ) -> list[str | None]:
        """Run many single-prompt completions in parallel (stdlib threads).

        Returns a list aligned with ``prompts`` (``None`` for a failed call).
        Requests are I/O-bound HTTP calls, so a thread pool gives a near-linear
        speed-up without any third-party async library. Progress is printed to
        stdout so it's obvious how far the batch has got. If the preflight has
        already determined the gateway is unreachable, returns all-``None``
        immediately instead of launching doomed threads.
        """
        n = len(prompts)
        if n == 0:
            return []
        if self._reachable is False:
            print(
                f"[judge] skipping {n} {label_prefix or 'request'}(s): "
                f"gateway marked unreachable by preflight.",
                flush=True,
            )
            return [None] * n

        n_workers = max(1, int(workers or self.workers))
        n_workers = min(n_workers, n)
        results: list[str | None] = [None] * n
        done = 0
        done_lock = threading.Lock()
        t0 = time.time()

        def _one(i: int) -> None:
            nonlocal done
            lab = f"{label_prefix}{i + 1}/{n}" if label_prefix else f"{i + 1}/{n}"
            results[i] = self.complete_text(
                prompts[i],
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                label=lab,
            )
            with done_lock:
                done += 1
                cur = done
            if progress_every and (cur % progress_every == 0 or cur == n):
                ok = sum(1 for r in results if r)
                rate = cur / max(1e-6, time.time() - t0)
                print(
                    f"[judge] {label_prefix or 'batch'} progress {cur}/{n} "
                    f"(ok={ok}, {rate:.1f} req/s)",
                    flush=True,
                )

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_one, range(n)))
        return results

    # -- parsing helpers ----------------------------------------------------

    @staticmethod
    def parse_json_object(content: str | None) -> dict | None:
        """Extract the first JSON object from a model reply (robust to prose).

        Tries the whole string first, then any ``{...}`` block. Returns the
        parsed dict or ``None`` when nothing parses.
        """
        if not content:
            return None
        for candidate in (content, *re.findall(r"\{.*?\}", content, flags=re.S)):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None
