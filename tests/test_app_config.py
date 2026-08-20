"""
Regression test for a real bug caught during Docker verification:
docker-compose.yml passes EMBED_MODEL/GEN_MODEL through as
"${EMBED_MODEL:-}" — which sets the env var to an EMPTY STRING when the
caller didn't override it, not "unset". app.py originally used
os.environ.get(key, default), which only falls back on a missing key, not
an empty one — so every container would have booted with an empty model
name. Fixed by switching to `os.environ.get(key) or default`; this test
locks that behavior in.
"""

import subprocess
import sys


def _read_config(env_overrides):
    """Runs app.py's config block in a clean subprocess with the given env
    vars, since app.py's constants are computed once at import time and
    this repo's test suite otherwise imports app.py exactly once per
    process (polluting every other test if we reloaded it in-process)."""
    code = (
        "import app; "
        "print(app.EMBED_MODEL); "
        "print(app.GEN_MODEL); "
        "print(app.EMBED_BACKEND); "
        "print(app.GEN_BACKEND)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env_overrides,
        capture_output=True,
        text=True,
        cwd=".",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    embed_model, gen_model, embed_backend, gen_backend = result.stdout.strip().splitlines()
    return embed_model, gen_model, embed_backend, gen_backend


def _base_env(**overrides):
    import os

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("EMBED_MODEL", "GEN_MODEL", "EMBED_BACKEND", "GEN_BACKEND")
    }
    env.update(overrides)
    return env


def test_empty_string_embed_model_falls_back_to_default():
    embed_model, _, _, _ = _read_config(_base_env(EMBED_MODEL=""))
    assert embed_model == "nomic-embed-text"


def test_empty_string_gen_model_falls_back_to_default():
    _, gen_model, _, _ = _read_config(_base_env(GEN_MODEL=""))
    assert gen_model == "mistral"


def test_empty_string_backend_falls_back_to_default():
    _, _, embed_backend, gen_backend = _read_config(_base_env(EMBED_BACKEND="", GEN_BACKEND=""))
    assert embed_backend == "ollama"
    assert gen_backend == "ollama"


def test_explicit_model_override_still_wins():
    embed_model, gen_model, _, _ = _read_config(_base_env(EMBED_MODEL="custom-embed", GEN_MODEL="custom-gen"))
    assert embed_model == "custom-embed"
    assert gen_model == "custom-gen"


def test_model_default_follows_backend_choice():
    _, gen_model, _, gen_backend = _read_config(_base_env(GEN_BACKEND="groq"))
    assert gen_backend == "groq"
    assert gen_model == "openai/gpt-oss-20b"
