import os
from typing import Any

import requests


def _load_token(token: str | None = None) -> str:
    tok = token or os.environ.get("GITHUB_TOKEN")
    if not tok:
        raise RuntimeError("GITHUB_TOKEN not configured")
    return tok


def create_issue(repo: str, title: str, body: str = "", token: str | None = None) -> dict[str, Any]:
    """Create a GitHub issue for ``repo`` using ``GITHUB_TOKEN``.

    Parameters
    ----------
    repo:
        Repository in ``owner/name`` form.
    title:
        Title for the new issue.
    body:
        Optional issue body.
    token:
        Personal access token override. If omitted, ``GITHUB_TOKEN`` is used.

    Returns
    -------
    dict
        Parsed JSON response from the GitHub API.
    """
    tok = _load_token(token)
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {tok}",
        "Accept": "application/vnd.github+json",
    }
    payload: dict[str, Any] = {"title": title}
    if body:
        payload["body"] = body
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()
