"""Project settings and credentials.

Values are looked up in this order (first hit wins):
  1. real environment variables
  2. the .env file in the project root (never committed - see .env.example)
  3. Colab Secrets (the key icon in the Colab sidebar), if running on Colab

Settings used by utils/github_sync.py:
  GITHUB_TOKEN   personal access token with write access to the repo
                 (fine-grained token: "Contents: Read and write")
  GITHUB_REPO    "owner/name", e.g. "durak-dev/ImageClassification"
  GITHUB_BRANCH  branch that receives the uploads (default "checkpoints";
                 it is created from the repo's default branch if missing)

This module never prints or returns the token except through get().
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')

DEFAULTS = {'GITHUB_BRANCH': 'checkpoints'}


def _parse_env_file(path):
    """Minimal .env parser: KEY=VALUE lines, '#' comments, optional 'export ',
    optional single/double quotes around the value."""
    values = {}
    if not os.path.isfile(path):
        return values
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            if line.startswith('export '):
                line = line[len('export '):]
            key, _, value = line.partition('=')
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
                value = value[1:-1]
            values[key.strip()] = value
    return values


def _colab_secret(name):
    """Read a Colab Secret; returns None when not on Colab / not set / no access."""
    try:
        from google.colab import userdata              # only exists on Colab
        return userdata.get(name)
    except Exception:
        return None


def get(name, default=None):
    """Look up a setting (see the module docstring for the order)."""
    value = os.environ.get(name)
    if not value:
        value = _parse_env_file(ENV_PATH).get(name)
    if not value:
        value = _colab_secret(name)
    return value or DEFAULTS.get(name, default)


def github_settings():
    """(token, repo, branch) - token/repo are None when not configured."""
    return get('GITHUB_TOKEN'), get('GITHUB_REPO'), get('GITHUB_BRANCH')


def github_ready():
    """True if a token and a repo ('owner/name') are configured."""
    token, repo, _ = github_settings()
    return bool(token) and bool(repo) and '/' in repo
