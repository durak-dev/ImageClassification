"""Upload / download project files through the GitHub REST API.

Used for checkpoints when the notebook runs somewhere without a persistent disk
(Google Colab): instead of leaving a file in the VM, it is committed to the repo.
Credentials come from utils/config.py (.env, environment, or Colab Secrets).

Several files are committed together in ONE commit (git data API), so a
checkpoint of multiple files is atomic. The GitHub API accepts files up to
100 MB; keep checkpoints well below that.
"""
import base64
import io
import os

import pandas as pd
import requests

from utils import config

API = 'https://api.github.com'
MAX_BYTES = 100 * 1024 * 1024
TIMEOUT = 60


def _settings():
    token, repo, branch = config.github_settings()
    if not (token and repo and '/' in repo):
        raise RuntimeError("GitHub is not configured: set GITHUB_TOKEN and GITHUB_REPO "
                           "(owner/name) in .env, the environment, or Colab Secrets.")
    return token, repo, branch


def _api(method, path, token, **kwargs):
    """One API call. Returns the response; raises with a clear message on errors
    (the token is never included in messages)."""
    headers = {'Authorization': f'Bearer {token}',
               'Accept': kwargs.pop('accept', 'application/vnd.github+json'),
               'X-GitHub-Api-Version': '2022-11-28'}
    return requests.request(method, API + path, headers=headers, timeout=TIMEOUT, **kwargs)


def _check(resp, what, ok=(200, 201)):
    if resp.status_code not in ok:
        try:
            detail = resp.json().get('message', '')
        except Exception:
            detail = resp.text[:200]
        hint = ''
        if resp.status_code in (401, 403):
            hint = ' (check that the token is valid and has write access to the repository)'
        elif resp.status_code == 404:
            hint = ' (check GITHUB_REPO, and that the token can access that repository)'
        raise RuntimeError(f"GitHub: {what} failed - HTTP {resp.status_code}: {detail}{hint}")
    return resp


def _ensure_branch(repo, branch, token):
    """Return the head commit sha of `branch`, creating the branch from the
    repository's default branch if it does not exist yet."""
    resp = _api('GET', f'/repos/{repo}/git/ref/heads/{branch}', token)
    if resp.status_code == 200:
        return resp.json()['object']['sha']
    if resp.status_code != 404:
        _check(resp, f"reading branch '{branch}'")
    default = _check(_api('GET', f'/repos/{repo}', token), 'reading repository').json()['default_branch']
    base = _check(_api('GET', f'/repos/{repo}/git/ref/heads/{default}', token),
                  f"reading default branch '{default}'").json()['object']['sha']
    _check(_api('POST', f'/repos/{repo}/git/refs', token,
                json={'ref': f'refs/heads/{branch}', 'sha': base}), f"creating branch '{branch}'")
    print(f"Created branch '{branch}' from '{default}'")
    return base


def push_files(files, message, branch=None):
    """Commit several files to the configured repo in one commit.

    files: {path_in_repo: bytes | str | local_file_path}. A str that is an
           existing file path is read from disk; any other str is uploaded as text.
    Returns the URL of the commit."""
    token, repo, default_branch = _settings()
    branch = branch or default_branch

    payload = {}
    for repo_path, content in files.items():
        repo_path = repo_path.replace('\\', '/').lstrip('/')
        if isinstance(content, str):
            if os.path.isfile(content):
                with open(content, 'rb') as f:
                    content = f.read()
            else:
                content = content.encode('utf-8')
        if len(content) > MAX_BYTES:
            raise ValueError(f"{repo_path} is {len(content) / 1e6:.0f} MB; the GitHub API limit is 100 MB")
        payload[repo_path] = content

    head = _ensure_branch(repo, branch, token)
    base_tree = _check(_api('GET', f'/repos/{repo}/git/commits/{head}', token),
                       'reading head commit').json()['tree']['sha']

    tree = []
    for repo_path, content in payload.items():
        blob = _check(_api('POST', f'/repos/{repo}/git/blobs', token,
                           json={'content': base64.b64encode(content).decode('ascii'),
                                 'encoding': 'base64'}), f'uploading {repo_path}').json()
        tree.append({'path': repo_path, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})

    new_tree = _check(_api('POST', f'/repos/{repo}/git/trees', token,
                           json={'base_tree': base_tree, 'tree': tree}), 'creating tree').json()['sha']
    commit = _check(_api('POST', f'/repos/{repo}/git/commits', token,
                         json={'message': message, 'tree': new_tree, 'parents': [head]}),
                    'creating commit').json()
    _check(_api('PATCH', f'/repos/{repo}/git/refs/heads/{branch}', token,
                json={'sha': commit['sha']}), f"updating branch '{branch}'")

    url = commit.get('html_url') or f'https://github.com/{repo}/commit/{commit["sha"]}'
    print(f"Uploaded {len(payload)} file(s) to {repo}@{branch}: {', '.join(payload)}")
    return url


def fetch_file(repo_path, branch=None):
    """Download one file from the configured repo/branch and return its bytes."""
    token, repo, default_branch = _settings()
    branch = branch or default_branch
    resp = _api('GET', f'/repos/{repo}/contents/{repo_path.lstrip("/")}', token,
                params={'ref': branch}, accept='application/vnd.github.raw+json')
    if resp.status_code == 404:
        raise FileNotFoundError(f"{repo_path} not found on {repo}@{branch}")
    return _check(resp, f'downloading {repo_path}').content


# --------------------------------------------------------------------------
# Convenience wrappers for dataframes
# --------------------------------------------------------------------------

def save_csv(df, repo_path, root, to_github=False, message=None, index=False):
    """Save a dataframe as a checkpoint.

    to_github=True : commit it to GitHub and keep NO local copy (for Colab).
    to_github=False: write <root>/<repo_path> on disk."""
    if to_github:
        csv = df.to_csv(index=index).encode('utf-8')
        push_files({repo_path: csv}, message or f'checkpoint: {repo_path}')
    else:
        local = os.path.join(root, repo_path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        df.to_csv(local, index=index)
        print(f"saved {len(df)} rows -> {local}")


def load_csv(repo_path, root, from_github=False, **read_kwargs):
    """Load a checkpoint written by save_csv (same flag value)."""
    local = os.path.join(root, repo_path)
    if from_github:
        try:
            return pd.read_csv(io.BytesIO(fetch_file(repo_path)), **read_kwargs)
        except FileNotFoundError as e:
            # not on the checkpoint branch: use the copy that came with the clone, if any
            if not os.path.isfile(local):
                raise
            print(f"{e} - using the file from the cloned repo instead")
    return pd.read_csv(local, **read_kwargs)
