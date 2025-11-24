# GitHub Repository Configuration

This directory contains configuration files and scripts for managing the GitHub repository metadata.

## Repository Description

The repository's "About" section should contain:

> **An AI-powered tool that generates algorithmic trading strategies using multiple frontier and open-source language models including GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Pro, CodeQwen, and CodeGemma.**

## How to Update the Description

### Option 1: GitHub Actions Workflow (Recommended)

1. Go to the "Actions" tab in the GitHub repository
2. Select "Update Repository Description" workflow
3. Click "Run workflow"
4. The workflow will automatically update the description and add relevant topics

### Option 2: Run the Shell Script

```bash
./.github/set-repository-description.sh
```

This script will use GitHub CLI to update the description. Make sure you're authenticated with `gh auth login`.

### Option 3: GitHub Web Interface

1. Go to the repository page
2. Click the ⚙️ (gear) icon next to "About" in the right sidebar
3. Paste the description in the "Description" field
4. Click "Save changes"

### Option 4: GitHub CLI (Manual)

```bash
gh repo edit AsutoshaNanda/llm-trading-strategy-generator \
  --description "An AI-powered tool that generates algorithmic trading strategies using multiple frontier and open-source language models including GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Pro, CodeQwen, and CodeGemma."
```

### Option 5: GitHub API

```bash
curl -X PATCH \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/AsutoshaNanda/llm-trading-strategy-generator \
  -d '{"description":"An AI-powered tool that generates algorithmic trading strategies using multiple frontier and open-source language models including GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Pro, CodeQwen, and CodeGemma."}'
```

## Suggested Topics

The following topics are recommended for better discoverability:

- `ai`
- `trading`
- `llm`
- `algorithmic-trading`
- `gpt-4`
- `claude`
- `gemini`
- `machine-learning`
- `finance`
- `python`
- `trading-strategies`
- `openai`
- `anthropic`
- `google-ai`
- `huggingface`
- `gradio`

You can add these via:
- The GitHub web interface (same ⚙️ menu)
- GitHub CLI: `gh repo edit --add-topic <topic-name>`
- The GitHub Actions workflow (which adds them automatically)

## Files in This Directory

- **`REPOSITORY_DESCRIPTION.md`**: Detailed documentation about the repository description
- **`set-repository-description.sh`**: Shell script to update the description using gh CLI
- **`workflows/update-repository-description.yml`**: GitHub Actions workflow for automated updates
- **`README.md`**: This file

## Notes

- Repository descriptions are stored in GitHub's database, not in repository files
- Changes require appropriate permissions (repo owner or admin)
- The description appears in the "About" section on the right sidebar of the repository page
- Topics help with repository discoverability in GitHub search
