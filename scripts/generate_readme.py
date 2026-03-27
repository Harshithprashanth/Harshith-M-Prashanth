#!/usr/bin/env python3
"""Generate and update the GitHub profile README dynamically.

This script fetches live data from GitHub, HuggingFace, and arXiv APIs,
reads configuration from config/profile_config.json, and regenerates
dynamic sections of the README.md file.

Usage:
    python scripts/generate_readme.py --config config/profile_config.json
    python scripts/generate_readme.py --config config/profile_config.json --dry-run
    python scripts/generate_readme.py --config config/profile_config.json --output README.md

Environment Variables:
    GITHUB_TOKEN: GitHub personal access token (optional, increases rate limit)
    GITHUB_USERNAME: GitHub username override
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"
HUGGINGFACE_API_BASE = "https://huggingface.co/api"
ARXIV_API_BASE = "http://export.arxiv.org/api/query"

DEFAULT_CONFIG_PATH = Path("config/profile_config.json")
DEFAULT_README_PATH = Path("README.md")


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


def fetch_github_stats(username: str, token: str | None = None) -> dict[str, Any]:
    """Fetch GitHub repository statistics for a given username.

    Args:
        username: GitHub username string.
        token: Optional GitHub personal access token for higher rate limits.
            When not provided the unauthenticated rate limit (60 req/hr) applies.

    Returns:
        Dictionary containing:
            - follower_count (int): Number of GitHub followers.
            - public_repo_count (int): Number of public repositories.
            - total_stars (int): Aggregate star count across all public repos.
            - top_repos (list[dict]): Top 5 repos by star count.

    Raises:
        requests.HTTPError: If the GitHub API returns a non-200 status code.
        ValueError: If username is empty or None.
    """
    if not username:
        raise ValueError("GitHub username must not be empty.")

    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Fetch user profile
    user_url = f"{GITHUB_API_BASE}/users/{username}"
    user_resp = requests.get(user_url, headers=headers, timeout=10)
    user_resp.raise_for_status()
    user_data = user_resp.json()

    # Fetch all public repos (paginated)
    repos: list[dict] = []
    page = 1
    while True:
        repos_url = (
            f"{GITHUB_API_BASE}/users/{username}/repos"
            f"?per_page=100&page={page}&sort=stars"
        )
        repos_resp = requests.get(repos_url, headers=headers, timeout=10)
        repos_resp.raise_for_status()
        page_repos = repos_resp.json()
        if not page_repos:
            break
        repos.extend(page_repos)
        if len(page_repos) < 100:
            break
        page += 1

    total_stars = sum(repo.get("stargazers_count", 0) for repo in repos)
    top_repos = sorted(repos, key=lambda r: r.get("stargazers_count", 0), reverse=True)[
        :5
    ]

    return {
        "follower_count": user_data.get("followers", 0),
        "public_repo_count": user_data.get("public_repos", 0),
        "total_stars": total_stars,
        "top_repos": [
            {
                "name": r["name"],
                "description": r.get("description", ""),
                "stars": r.get("stargazers_count", 0),
                "language": r.get("language", ""),
                "url": r.get("html_url", ""),
            }
            for r in top_repos
        ],
    }


# ---------------------------------------------------------------------------
# HuggingFace API helpers
# ---------------------------------------------------------------------------


def fetch_huggingface_stats(username: str) -> dict[str, Any]:
    """Fetch model and dataset statistics from HuggingFace for a given user.

    Args:
        username: HuggingFace username or organisation name.

    Returns:
        Dictionary containing:
            - model_count (int): Number of public models.
            - dataset_count (int): Number of public datasets.
            - total_downloads (int): Aggregate download count across all models.
            - top_models (list[dict]): Top 3 models by download count.

    Note:
        HuggingFace API does not require authentication for public data.
        Returns zeroed statistics gracefully if the API is unreachable.
    """
    stats: dict[str, Any] = {
        "model_count": 0,
        "dataset_count": 0,
        "total_downloads": 0,
        "top_models": [],
    }

    try:
        models_url = f"{HUGGINGFACE_API_BASE}/models?author={username}&limit=100"
        resp = requests.get(models_url, timeout=10)
        resp.raise_for_status()
        models = resp.json()

        stats["model_count"] = len(models)
        stats["total_downloads"] = sum(m.get("downloads", 0) for m in models)
        top_models = sorted(models, key=lambda m: m.get("downloads", 0), reverse=True)[
            :3
        ]
        stats["top_models"] = [
            {
                "id": m.get("id", ""),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
            }
            for m in top_models
        ]

        datasets_url = f"{HUGGINGFACE_API_BASE}/datasets?author={username}&limit=100"
        ds_resp = requests.get(datasets_url, timeout=10)
        ds_resp.raise_for_status()
        stats["dataset_count"] = len(ds_resp.json())

    except requests.RequestException as exc:
        logger.warning(f"Could not fetch HuggingFace stats: {exc}")

    return stats


# ---------------------------------------------------------------------------
# arXiv API helpers
# ---------------------------------------------------------------------------


def fetch_arxiv_recent_papers(
    search_query: str, max_results: int = 5
) -> list[dict[str, Any]]:
    """Fetch recent papers from arXiv matching the given search query.

    Args:
        search_query: arXiv API search query string, e.g.,
            "au:Prashanth_H" or "ti:LLM+signal+processing".
        max_results: Maximum number of results to return (default 5).

    Returns:
        List of paper dictionaries, each containing:
            - title (str): Paper title.
            - authors (list[str]): List of author names.
            - abstract (str): Paper abstract (truncated to 300 chars).
            - published (str): Publication date (ISO format).
            - arxiv_id (str): arXiv identifier.
            - url (str): arXiv abstract page URL.

    Note:
        Returns an empty list gracefully if the arXiv API is unreachable.
    """
    papers: list[dict[str, Any]] = []

    try:
        import xml.etree.ElementTree as ET

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = requests.get(ARXIV_API_BASE, params=params, timeout=15)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)
            id_el = entry.find("atom:id", ns)
            authors = [
                a.find("atom:name", ns).text or ""
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ]

            if title_el is None or id_el is None:
                continue

            arxiv_url = id_el.text or ""
            arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""
            abstract = (summary_el.text or "").strip()
            if len(abstract) > 300:
                abstract = abstract[:297] + "..."

            papers.append(
                {
                    "title": (title_el.text or "").strip().replace("\n", " "),
                    "authors": authors,
                    "abstract": abstract,
                    "published": (published_el.text or "")[:10] if published_el is not None else "",
                    "arxiv_id": arxiv_id,
                    "url": arxiv_url,
                }
            )

    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not fetch arXiv papers: {exc}")

    return papers


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def generate_llm_skills_section(config: dict[str, Any]) -> str:
    """Generate the LLM & LLMOps expertise section from config data.

    Args:
        config: Parsed profile configuration dictionary from profile_config.json.

    Returns:
        Markdown string for the LLM skills section.
    """
    llm_skills = config.get("llmSkills", {})
    frameworks = llm_skills.get("frameworks", [])
    evaluation = llm_skills.get("evaluationFrameworks", [])

    lines: list[str] = [
        "## ⚙️ LLM & LLMOps Expertise\n",
        "| Category | Tools & Frameworks | Proficiency |",
        "|---|---|---|",
        f"| LLM Frameworks | {', '.join(f['name'] for f in frameworks[:3])} | ⭐⭐⭐⭐⭐ |",
        "| Fine-tuning | LoRA, QLoRA, PEFT, TRL | ⭐⭐⭐⭐⭐ |",
        "| RAG Systems | ChromaDB, Pinecone, FAISS, HyDE | ⭐⭐⭐⭐⭐ |",
        "| LLM APIs | OpenAI, Anthropic, Gemini, Cohere | ⭐⭐⭐⭐⭐ |",
        "| Local LLMs | Ollama, vLLM, llama.cpp | ⭐⭐⭐⭐ |",
        "| Monitoring | LangSmith, W&B, MLflow | ⭐⭐⭐⭐ |",
        f"| Evaluation | {', '.join(f['name'] for f in evaluation[:3])} | ⭐⭐⭐⭐ |",
        "| Deployment | FastAPI, BentoML, Docker | ⭐⭐⭐⭐ |",
        "| Agents | ReAct, Tool Use, LangGraph Agents | ⭐⭐⭐⭐ |",
        "| Guardrails | Guardrails AI, NeMo Guardrails | ⭐⭐⭐ |",
        "",
    ]
    return "\n".join(lines)


def generate_projects_section(config: dict[str, Any]) -> str:
    """Generate the projects showcase section from config data.

    Args:
        config: Parsed profile configuration dictionary from profile_config.json.

    Returns:
        Markdown string for the projects section with collapsible details.
    """
    categories = config.get("projects", {}).get("categories", [])
    lines: list[str] = ["## 🚀 Featured Projects\n"]

    for category in categories:
        emoji = category.get("emoji", "")
        title = category.get("title", "")
        lines.append(f"<details>\n<summary><b>{emoji} {title}</b></summary>\n")

        for project in category.get("projects", []):
            p_title = project.get("title", "Unknown")
            domain = project.get("domain", "")
            metrics = project.get("metrics", {})
            tech = project.get("tech", [])

            lines.append(f"\n### {p_title}")
            if domain:
                lines.append(f"**Domain**: {domain}\n")
            if metrics:
                metric_strs = [f"{k}: {v}" for k, v in metrics.items()]
                lines.append(f"**Results**: {' · '.join(metric_strs)}\n")
            if tech:
                lines.append(f"**Tech**: {' · '.join(tech)}\n")

        lines.append("</details>\n")

    return "\n".join(lines)


def generate_github_stats_section(stats: dict[str, Any], config: dict[str, Any]) -> str:
    """Generate the GitHub statistics section.

    Args:
        stats: GitHub statistics dictionary from fetch_github_stats().
        config: Parsed profile configuration dictionary.

    Returns:
        Markdown string for the GitHub stats section.
    """
    username = config.get("profile", {}).get("username", "Harshithprashanth")
    theme = config.get("theme", {}).get("statsTheme", "tokyonight")

    lines: list[str] = [
        "## 📈 GitHub Stats\n",
        '<div align="center">\n',
        f"![GitHub Stats](https://github-readme-stats.vercel.app/api?username={username}"
        f"&show_icons=true&theme={theme}&hide_border=true&count_private=true)\n",
        f"![Top Languages](https://github-readme-stats.vercel.app/api/top-langs/"
        f"?username={username}&layout=compact&theme={theme}&hide_border=true)\n",
        f"![GitHub Streak](https://streak-stats.demolab.com?user={username}"
        f"&theme={theme}&hide_border=true)\n",
        "</div>\n",
        f"> 🌟 **{stats.get('total_stars', 0)}** total stars across "
        f"**{stats.get('public_repo_count', 0)}** public repositories | "
        f"**{stats.get('follower_count', 0)}** followers\n",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# README updater
# ---------------------------------------------------------------------------


def update_readme(sections: dict[str, str], readme_path: Path) -> None:
    """Write updated sections into the README file.

    This function replaces marked dynamic sections in the README with freshly
    generated content. Static sections are preserved unchanged.

    The README may contain marker comments of the form:
        <!-- BEGIN_SECTION:section_name -->
        ... dynamic content ...
        <!-- END_SECTION:section_name -->

    If no markers are found for a section, the section is appended at the end
    of the file.

    Args:
        sections: Mapping of section names to their generated Markdown content.
        readme_path: Path to the README.md file to update.

    Returns:
        None. Writes changes to readme_path in place.
    """
    if not readme_path.exists():
        logger.error(f"README not found at {readme_path}")
        return

    content = readme_path.read_text(encoding="utf-8")

    for section_name, section_content in sections.items():
        begin_marker = f"<!-- BEGIN_SECTION:{section_name} -->"
        end_marker = f"<!-- END_SECTION:{section_name} -->"

        if begin_marker in content and end_marker in content:
            # Replace between markers
            start_idx = content.index(begin_marker) + len(begin_marker)
            end_idx = content.index(end_marker)
            content = (
                content[: start_idx]
                + f"\n\n{section_content}\n\n"
                + content[end_idx:]
            )
            logger.info(f"Updated section: {section_name}")
        else:
            logger.debug(
                f"No markers found for section '{section_name}'. Section skipped."
            )

    # Update last-updated badge
    today = datetime.now(tz=timezone.utc).strftime("%B_%Y")
    content = content.replace(
        "Last_Updated-March_2026", f"Last_Updated-{today}"
    )

    readme_path.write_text(content, encoding="utf-8")
    logger.success(f"README updated at {readme_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional list of argument strings (defaults to sys.argv[1:]).

    Returns:
        Parsed argparse Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Auto-generate dynamic sections of the GitHub profile README.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to profile_config.json (default: config/profile_config.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_README_PATH,
        help="Path to output README.md (default: README.md)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated sections to stdout without writing to disk.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the README generator.

    Args:
        argv: Optional list of argument strings (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    args = parse_args(argv)

    # Configure logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=log_level, colorize=True)

    # Load configuration
    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1

    try:
        with args.config.open(encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid JSON in config file: {exc}")
        return 1

    profile = config.get("profile", {})
    username = os.environ.get("GITHUB_USERNAME") or profile.get("username", "Harshithprashanth")
    github_token = os.environ.get("GITHUB_TOKEN")

    # Fetch live data
    logger.info(f"Fetching GitHub stats for user: {username}")
    try:
        github_stats = fetch_github_stats(username, token=github_token)
        logger.info(
            f"GitHub stats fetched: {github_stats['public_repo_count']} repos, "
            f"{github_stats['total_stars']} stars"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"GitHub stats fetch failed: {exc}. Using empty stats.")
        github_stats = {"follower_count": 0, "public_repo_count": 0, "total_stars": 0, "top_repos": []}

    hf_username = (
        profile.get("urls", {}).get("huggingface", "").split("/")[-1] or username
    )
    logger.info(f"Fetching HuggingFace stats for user: {hf_username}")
    hf_stats = fetch_huggingface_stats(hf_username)
    logger.info(f"HuggingFace models: {hf_stats['model_count']}")

    logger.info("Fetching recent arXiv papers...")
    arxiv_query = f"au:{username.replace('github', '')}"
    recent_papers = fetch_arxiv_recent_papers(arxiv_query, max_results=5)
    logger.info(f"arXiv papers found: {len(recent_papers)}")

    # Generate dynamic sections
    sections: dict[str, str] = {
        "llm_expertise": generate_llm_skills_section(config),
        "projects": generate_projects_section(config),
        "github_stats": generate_github_stats_section(github_stats, config),
    }

    if args.dry_run:
        logger.info("Dry run — printing generated sections to stdout:")
        for name, content in sections.items():
            print(f"\n{'='*60}")
            print(f"  SECTION: {name}")
            print(f"{'='*60}")
            print(content)
        return 0

    update_readme(sections, args.output)
    logger.success("README generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
