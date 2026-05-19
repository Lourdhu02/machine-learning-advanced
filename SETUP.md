# Setup (one time)

This course needs almost nothing — NumPy, matplotlib, Jupyter, and PyTorch only from module 10 onward. Each module folder has its own `requirements.txt` so you install only what you need.

---

## 1. Python 3.11+

Verify:
```powershell
python --version
```

If missing or older, install from https://www.python.org/downloads/. On Windows, tick "Add Python to PATH".

Upgrade pip:
```powershell
python -m pip install --upgrade pip
```

---

## 2. VS Code + extensions

Install VS Code: https://code.visualstudio.com/

Extensions:

| Extension | Why |
|---|---|
| `ms-python.python` | Python language support, venv detection |
| `ms-python.vscode-pylance` | Fast type checker |
| `ms-toolsai.jupyter` | Notebooks inside VS Code |
| `bierner.markdown-mermaid` | Render Mermaid mind-maps in the README preview |

---

## 3. Per-module venv (the pattern)

Every module is self-contained. Memorize this:

```powershell
cd course\01-linear-regression
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then in VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → pick `.venv`.

To leave: `deactivate`.

> Why per-module? Modules 10+ pull in PyTorch (~2 GB). Classical modules don't. Isolation keeps each module fast to install.

---

## 4. Regenerating diagrams

Every diagram in this course is a tiny matplotlib script that lives next to the README. To regenerate:

```powershell
python diagram_fit.py
```

The `.png` files are committed so they render on GitHub. Edit the script and re-run to change them.

---

## 5. Git + GitHub

If not configured:

```powershell
git --version
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## 6. Verify

Run module 00:

```powershell
cd course\00-math-foundations
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python from_scratch.py
```

If the gradient and eigendecomposition checks at the end print `OK`, you're set.

---

## Troubleshooting

- **`Activate.ps1` blocked**: run once as your user:
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
- **Mermaid not rendering on GitHub**: the fenced block language must be exactly ` ```mermaid `.
- **`pip install` SSL errors**: `pip install --upgrade certifi`.
