# Contributing to Harshith Prashanth's GitHub Profile Repository

Thank you for your interest in contributing to this repository. This project
serves as a professional portfolio and research showcase. Contributions that
improve accuracy, clarity, or the quality of the research documentation are
warmly welcomed.

---

## 📋 Table of Contents

- [How to Contribute](#how-to-contribute)
- [Code Style Guidelines](#code-style-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Templates](#issue-templates)
- [Documentation Standards](#documentation-standards)

---

## 🤝 How to Contribute

### Types of Contributions Welcome

1. **Bug fixes** — Incorrect information, broken links, or rendering issues in Markdown files
2. **Improvements to scripts** — Enhancements to `scripts/generate_readme.py` or other automation scripts
3. **Documentation** — Clarifications, additions to research documentation
4. **Typo / grammar corrections** — Especially in formal documents (research statement, academic CV)
5. **Feature additions** — New sections to profile documents, additional automation features

### Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Harshith-M-Prashanth.git
   cd Harshith-M-Prashanth
   ```
3. Create a feature branch:
   ```bash
   git checkout -b fix/broken-links
   # or
   git checkout -b feature/add-waka-stats
   ```
4. Make your changes
5. Test your changes (see below)
6. Submit a pull request

---

## 🎨 Code Style Guidelines

### Python (scripts/)

All Python code must conform to:

- **PEP 8** style guide (enforced via `flake8` and `black`)
- **Type hints** for all function signatures
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Logging**: Use the `loguru` library for logging; no bare `print()` statements in production code
- **Error handling**: All external API calls must include proper exception handling

Example:
```python
def fetch_github_stats(username: str, token: str | None = None) -> dict:
    """Fetch GitHub repository statistics for a given username.

    Args:
        username: GitHub username string.
        token: Optional GitHub personal access token for higher rate limits.

    Returns:
        Dictionary containing follower count, public repo count, and
        total star count.

    Raises:
        requests.HTTPError: If the GitHub API returns a non-200 status code.
        ValueError: If username is empty or None.
    """
    ...
```

- Maximum line length: **88 characters** (black default)
- Use f-strings for string formatting (not `.format()` or `%`)
- All files must include a module-level docstring

### Markdown

- Use ATX-style headings (`#`, `##`, `###`)
- Include a Table of Contents for files with 3+ major sections
- Use reference-style links for repeated URLs
- Wrap lines at 120 characters for prose paragraphs
- Code blocks must specify the language for syntax highlighting
- Include the `Last Updated: [Date]` footer on all documentation files

### JSON (config/)

- 2-space indentation
- All keys in camelCase
- Include descriptive comments where JSON format permits (use README files alongside JSON)

---

## 🔄 Pull Request Process

1. Ensure your branch is up to date with `main` before submitting:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. Run linting and tests:
   ```bash
   pip install -r requirements.txt
   flake8 scripts/ --max-line-length 88
   black --check scripts/
   python -m pytest tests/ -v  # if tests exist
   ```

3. Write a clear PR description including:
   - **What** changed
   - **Why** the change is necessary or beneficial
   - **How** to verify the change

4. Link any related issues with `Closes #issue-number` in the PR body

5. PRs require review and approval before merging

6. Squash commits before merging to maintain a clean history

---

## 📝 Issue Templates

### Bug Report

```markdown
**Description**: [Clear description of the bug]
**File affected**: [e.g., README.md, scripts/generate_readme.py]
**Steps to reproduce**: [Numbered steps]
**Expected behaviour**: [What should happen]
**Actual behaviour**: [What actually happens]
**Environment**: [OS, Python version, relevant package versions]
```

### Feature Request

```markdown
**Summary**: [One-line summary of the feature]
**Motivation**: [Why is this feature needed?]
**Proposed implementation**: [Your suggested approach]
**Alternatives considered**: [Other approaches you evaluated]
```

---

## 📚 Documentation Standards

- All research documents should maintain formal, academic English
- Technical claims must be accurate and, where possible, include citations
- Placeholder sections should use the format `[PLACEHOLDER: description]`
- Version history or changelog entries are encouraged for significant document updates

---

## 📜 Code of Conduct

By contributing to this repository, you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

---

*Last Updated: March 2026*
