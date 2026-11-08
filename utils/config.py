"""Project settings and credentials.

Values are looked up in this order (first hit wins):
  1. real environment variables
  2. the .env file in the project root (never committed - see .env.example)
  3. Colab Secrets (the key icon in the Colab sidebar), if running on Colab

Settings used by utils/github_sync.py:
  GITHUB_TOKEN   personal access token with write access to the repo
                 (fine-grained token: "Contents: Read and write")
  GITHUB_REPO    "owner/name", e.g. "durak-dev/ImageClassification"
                 (optional: defaults to the repo the project was cloned from)
  GITHUB_BRANCH  branch that receives the uploads (default "checkpoints";
                 it is created from the repo's default branch if missing)

This module never prints or returns the token except through get().
"""
import os
import re
import subprocess
import getpass

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


def _git_origin_repo():
    """'owner/name' parsed from the clone's `origin` remote (works for https and
    ssh urls; any credentials embedded in the url are ignored). None if unknown."""
    try:
        url = subprocess.run(['git', '-C', PROJECT_ROOT, 'remote', 'get-url', 'origin'],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None
    match = re.search(r'github\.com[:/]+([^/\s]+)/([^/\s]+?)(?:\.git)?/?$', url)
    return f'{match.group(1)}/{match.group(2)}' if match else None


def get(name, default=None):
    """Look up a setting (see the module docstring for the order)."""
    value = os.environ.get(name)
    if not value:
        value = _parse_env_file(ENV_PATH).get(name)
    if not value:
        value = _colab_secret(name)
    if not value and name == 'GITHUB_REPO':
        value = _git_origin_repo()                 # the repo this notebook was cloned from
    return value or DEFAULTS.get(name, default)


def github_settings():
    """(token, repo, branch) - token/repo are None when not configured."""
    return get('GITHUB_TOKEN'), get('GITHUB_REPO'), get('GITHUB_BRANCH')


def github_ready():
    """True if a token and a repo ('owner/name') are configured."""
    token, repo, _ = github_settings()
    return bool(token) and bool(repo) and '/' in repo


def ensure_github(prompt=True):
    """Make sure a GitHub token is available. If none is configured and
    prompt=True, ask for it with a hidden input box and keep it in memory for
    this session only (nothing is written to disk). Leave the box empty to skip.
    Returns github_ready()."""
    if not github_ready() and prompt:
        token = getpass.getpass('GitHub token (hidden, kept in memory only; empty = no upload): ').strip()
        if token:
            os.environ['GITHUB_TOKEN'] = token
    return github_ready()
