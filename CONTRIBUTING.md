# Contributing to GastricPath AI

Thank you for your interest in contributing to **GastricPath AI**, an open computational pathology research platform.

---

## 1. Code of Conduct & Scientific Rigor

We adhere to the highest standards of scientific reproducibility, computational transparency, and academic integrity:
- **No Fabricated Data:** Never generate or commit synthetic patient records, artificial annotations, or false benchmark results.
- **Reproducible Experiments:** All machine learning experiments must document random seeds, exact package dependencies, and hardware execution configurations.
- **Honest Metric Reporting:** Always report both positive and negative findings, class imbalances, and performance limitations.

---

## 2. Development Workflow

### Step 1: Fork and Clone
```bash
git clone https://github.com/DevPandey25/Gastric-Cancer-Path-Detection.git
cd Gastric-Cancer-Path-Detection
```

### Step 2: Virtual Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 3: Run the Test Suite
Before introducing changes, ensure all existing tests pass:
```powershell
python -m pytest tests/ -q
```

### Step 4: Branching & Commits
- Use descriptive branch names: `feature/new-encoder`, `fix/subnav-binding`, `docs/citation-update`.
- Write clear, conventional commit messages: `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`.

### Step 5: Submitting a Pull Request
1. Ensure `python -m pytest tests/` passes with 0 failures.
2. Confirm that `.env`, local `.db` files, and raw images are not staged.
3. Open a Pull Request describing your research motivation, methodology, and verification steps.

