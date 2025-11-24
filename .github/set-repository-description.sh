#!/bin/bash

# Script to set the GitHub repository description
# This script should be run by the repository owner with appropriate permissions

REPO_OWNER="AsutoshaNanda"
REPO_NAME="llm-trading-strategy-generator"
DESCRIPTION="An AI-powered tool that generates algorithmic trading strategies using multiple frontier and open-source language models including GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Pro, CodeQwen, and CodeGemma."

echo "Setting repository description for ${REPO_OWNER}/${REPO_NAME}..."
echo ""
echo "Description to be set:"
echo "  ${DESCRIPTION}"
echo ""

# Check if gh CLI is available
if command -v gh &> /dev/null; then
    echo "Using GitHub CLI (gh)..."
    gh repo edit "${REPO_OWNER}/${REPO_NAME}" --description "${DESCRIPTION}"
    
    if [ $? -eq 0 ]; then
        echo "✓ Repository description updated successfully!"
        echo ""
        echo "You can also add topics using:"
        echo "  gh repo edit ${REPO_OWNER}/${REPO_NAME} --add-topic ai,trading,llm,algorithmic-trading,gpt-4,claude,gemini,machine-learning,finance,python"
    else
        echo "✗ Failed to update repository description using gh CLI"
        echo "Please ensure you are authenticated with: gh auth login"
    fi
else
    echo "GitHub CLI (gh) is not installed."
    echo ""
    echo "Please install it from: https://cli.github.com/"
    echo ""
    echo "Or update manually via:"
    echo "  1. Go to https://github.com/${REPO_OWNER}/${REPO_NAME}"
    echo "  2. Click the ⚙️ icon next to 'About' on the right sidebar"
    echo "  3. Paste the description and save"
fi
