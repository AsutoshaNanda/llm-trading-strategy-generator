# Quick Start: Update Repository Description

## ⚡ Fastest Method - GitHub Actions (Recommended)

1. Go to: https://github.com/AsutoshaNanda/llm-trading-strategy-generator/actions
2. Click on "Update Repository Description" in the left sidebar
3. Click the "Run workflow" button (on the right)
4. Click the green "Run workflow" button in the dropdown
5. Done! ✅

The workflow will automatically:
- Update the repository description
- Add relevant topics (ai, trading, llm, etc.)

---

## 🔧 Alternative: Manual Update via Web UI

1. Go to: https://github.com/AsutoshaNanda/llm-trading-strategy-generator
2. Look at the right sidebar for the "About" section
3. Click the ⚙️ (gear) icon next to "About"
4. In the "Description" field, paste:
   ```
   An AI-powered tool that generates algorithmic trading strategies using multiple frontier and open-source language models including GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Pro, CodeQwen, and CodeGemma.
   ```
5. (Optional) Add topics: `ai`, `trading`, `llm`, `algorithmic-trading`, `gpt-4`, `claude`, `gemini`, `machine-learning`, `finance`, `python`
6. Click "Save changes"
7. Done! ✅

---

## 💻 Alternative: Command Line

If you have GitHub CLI installed and authenticated:

```bash
gh repo edit AsutoshaNanda/llm-trading-strategy-generator \
  --description "An AI-powered tool that generates algorithmic trading strategies using multiple frontier and open-source language models including GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Pro, CodeQwen, and CodeGemma."
```

Or run the provided script:

```bash
./.github/set-repository-description.sh
```

---

## 📝 What Will Change?

**Before**: Repository has no description in the About section

**After**: The About section will show:
> An AI-powered tool that generates algorithmic trading strategies using multiple frontier and open-source language models including GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Pro, CodeQwen, and CodeGemma.

Plus relevant topics for better discoverability!

---

## ❓ Need Help?

See `.github/README.md` for detailed documentation and all available options.
