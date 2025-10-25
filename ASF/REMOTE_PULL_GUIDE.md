# Remote Server Pull Guide

Complete guide for pulling the latest SDE sampling code on your remote server.

---

## 🚀 Quick Pull Commands

### Option 1: Simple Pull (Recommended)

```bash
# SSH into your remote server
ssh user@your-server.com

# Navigate to project directory
cd /path/to/NFDiffusion

# Pull latest changes
git pull origin v2_local_implicit_unet

# Verify new files exist
ls -lh ASF/eval_sde*
```

### Option 2: Safe Pull with Stash (If You Have Local Changes)

```bash
# SSH into remote server
ssh user@your-server.com
cd /path/to/NFDiffusion

# Stash any local changes
git stash

# Pull latest changes
git pull origin v2_local_implicit_unet

# Reapply your local changes (if needed)
git stash pop
```

### Option 3: Fresh Clone (Clean Start)

```bash
# SSH into remote server
ssh user@your-server.com

# Backup old directory (optional)
mv NFDiffusion NFDiffusion_backup

# Fresh clone
git clone https://github.com/colpark/NFDiffusion.git
cd NFDiffusion

# Checkout the correct branch
git checkout v2_local_implicit_unet

# Verify
git branch
ls -lh ASF/eval_sde*
```

---

## 📋 Complete Workflow Script

Save this as `remote_setup.sh` on your remote server:

```bash
#!/bin/bash
# remote_setup.sh - Setup script for remote server

set -e  # Exit on error

echo "============================================================"
echo "NFDiffusion SDE Sampling Setup"
echo "============================================================"

# Configuration
PROJECT_DIR="${HOME}/NFDiffusion"
BRANCH="v2_local_implicit_unet"
REPO_URL="https://github.com/colpark/NFDiffusion.git"

# Check if directory exists
if [ -d "$PROJECT_DIR" ]; then
    echo "📁 Project directory exists, updating..."
    cd "$PROJECT_DIR"

    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        echo "⚠️  Uncommitted changes detected, stashing..."
        git stash
        STASHED=true
    fi

    # Fetch and pull
    echo "📥 Fetching latest changes..."
    git fetch origin

    echo "🔄 Pulling branch: $BRANCH"
    git pull origin "$BRANCH"

    # Restore stashed changes
    if [ "$STASHED" = true ]; then
        echo "♻️  Restoring stashed changes..."
        git stash pop
    fi
else
    echo "📦 Cloning repository..."
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    git checkout "$BRANCH"
fi

# Verify new files
echo ""
echo "✅ Verifying SDE files..."
if [ -f "ASF/eval_sde.sh" ] && [ -f "ASF/eval_sde_multiscale.py" ]; then
    echo "✓ SDE evaluation files found"
    chmod +x ASF/eval_sde.sh
    echo "✓ Made eval_sde.sh executable"
else
    echo "❌ Error: SDE files not found!"
    exit 1
fi

# Check Python dependencies
echo ""
echo "🐍 Checking Python environment..."
python3 -c "import torch, numpy, matplotlib, sklearn" 2>/dev/null && echo "✓ Dependencies OK" || echo "⚠️  Install: pip install torch numpy matplotlib scikit-image"

# Display summary
echo ""
echo "============================================================"
echo "✅ Setup Complete!"
echo "============================================================"
echo "Project directory: $PROJECT_DIR"
echo "Current branch: $(git branch --show-current)"
echo "Latest commit: $(git log -1 --oneline)"
echo ""
echo "📊 New files added:"
echo "  - ASF/eval_sde.sh (executable)"
echo "  - ASF/eval_sde_multiscale.py"
echo "  - ASF/README_SDE.md"
echo "  - ASF/QUICKSTART_SDE.md"
echo ""
echo "🚀 Ready to run:"
echo "  cd $PROJECT_DIR/ASF"
echo "  ./eval_sde.sh"
echo "============================================================"
```

**Usage on remote server**:

```bash
# Upload and run setup script
scp remote_setup.sh user@server:~
ssh user@server "bash ~/remote_setup.sh"
```

---

## 🔧 Step-by-Step Manual Process

### Step 1: Check Current Status

```bash
ssh user@your-server.com
cd /path/to/NFDiffusion

# Check current branch
git branch

# Check current commit
git log -1 --oneline

# Check for local changes
git status
```

### Step 2: Backup Local Changes (If Needed)

```bash
# If you have uncommitted changes
git stash save "backup before SDE pull"

# Or create a backup branch
git checkout -b backup-$(date +%Y%m%d)
git checkout v2_local_implicit_unet
```

### Step 3: Fetch and Pull

```bash
# Fetch latest changes
git fetch origin

# See what will be pulled
git log HEAD..origin/v2_local_implicit_unet --oneline

# Pull the changes
git pull origin v2_local_implicit_unet
```

### Step 4: Verify New Files

```bash
# List new SDE files
ls -lh ASF/eval_sde*

# Output should show:
# -rwxr-xr-x  1 user  group  3.2K  eval_sde.sh
# -rw-r--r--  1 user  group   28K  eval_sde_multiscale.py

# Check if executable
[ -x ASF/eval_sde.sh ] && echo "✓ Executable" || chmod +x ASF/eval_sde.sh
```

### Step 5: Test Installation

```bash
cd ASF

# Quick verification
python3 -c "from train_mamba_standalone import sde_sample, ddim_sample; print('✓ Imports OK')"

# Check help
python3 eval_sde_multiscale.py --help
```

---

## 🐳 Docker/Container Setup

If running in Docker/Singularity:

### Docker

```bash
# SSH into server
ssh user@server

# Enter container
docker exec -it your-container bash

# Inside container
cd /workspace/NFDiffusion
git pull origin v2_local_implicit_unet
cd ASF
./eval_sde.sh
```

### Singularity

```bash
# SSH into server
ssh user@server

# Shell into container
singularity shell --nv /path/to/container.sif

# Inside container
cd /workspace/NFDiffusion
git pull origin v2_local_implicit_unet
cd ASF
./eval_sde.sh
```

---

## 🔐 SSH Key Setup (For Smoother Pulls)

If you need to set up SSH keys for GitHub:

### On Remote Server

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub
```

### On GitHub

1. Go to https://github.com/settings/keys
2. Click "New SSH key"
3. Paste your public key
4. Save

### Update Git Remote to Use SSH

```bash
cd /path/to/NFDiffusion

# Check current remote
git remote -v

# If using HTTPS, switch to SSH
git remote set-url origin git@github.com:colpark/NFDiffusion.git

# Verify
git remote -v
```

---

## 📦 Alternative: Direct File Download

If git isn't working, download files directly:

```bash
# On remote server
cd /path/to/NFDiffusion/ASF

# Download SDE evaluation script
wget https://raw.githubusercontent.com/colpark/NFDiffusion/v2_local_implicit_unet/ASF/eval_sde_multiscale.py

# Download bash runner
wget https://raw.githubusercontent.com/colpark/NFDiffusion/v2_local_implicit_unet/ASF/eval_sde.sh
chmod +x eval_sde.sh

# Download updated training script
wget https://raw.githubusercontent.com/colpark/NFDiffusion/v2_local_implicit_unet/ASF/train_mamba_standalone.py

# Download documentation
wget https://raw.githubusercontent.com/colpark/NFDiffusion/v2_local_implicit_unet/ASF/README_SDE.md
wget https://raw.githubusercontent.com/colpark/NFDiffusion/v2_local_implicit_unet/ASF/QUICKSTART_SDE.md

# Verify
ls -lh eval_sde*
```

---

## 🧪 Post-Pull Verification

### Quick Test

```bash
cd ASF

# Check Python can import new functions
python3 << EOF
from train_mamba_standalone import sde_sample, ddim_sample
print("✓ SDE sampler imported successfully")
print("✓ DDIM sampler imported successfully")
EOF

# Check bash script
bash -n eval_sde.sh && echo "✓ Bash script syntax OK"

# Quick dry-run
python3 eval_sde_multiscale.py --help
```

### Full Verification

```bash
cd ASF

# Run minimal test (requires trained checkpoint)
if [ -f "checkpoints_mamba/mamba_best.pth" ]; then
    echo "Running quick test with 2 samples..."
    NUM_SAMPLES=2 SAMPLERS="sde" ./eval_sde.sh
else
    echo "⚠️  No checkpoint found, skipping test"
    echo "Train model first with: ./run_mamba_training.sh"
fi
```

---

## 🐛 Troubleshooting

### Problem: "Branch not found"

```bash
# Update remote branches
git fetch origin

# List all branches
git branch -a

# Checkout correct branch
git checkout v2_local_implicit_unet
```

### Problem: "Merge conflict"

```bash
# See conflicting files
git status

# Option 1: Keep remote version (discard local changes)
git reset --hard origin/v2_local_implicit_unet

# Option 2: Keep local changes
git merge --strategy-option theirs origin/v2_local_implicit_unet
```

### Problem: "Permission denied"

```bash
# Fix SSH permissions
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

# Or use HTTPS instead
git remote set-url origin https://github.com/colpark/NFDiffusion.git
```

### Problem: "Files not executable"

```bash
cd ASF

# Make scripts executable
chmod +x eval_sde.sh
chmod +x eval_superres.sh
chmod +x run_mamba_training.sh

# Verify
ls -lh *.sh
```

### Problem: "Python import errors"

```bash
# Check if in correct directory
pwd  # Should be in ASF/

# Add parent to path (if needed)
export PYTHONPATH="${PYTHONPATH}:$(dirname $(pwd))"

# Or run from ASF directory
cd /path/to/NFDiffusion/ASF
python3 eval_sde_multiscale.py --help
```

---

## 📊 One-Liner Commands

### Just Pull and Run

```bash
ssh user@server "cd ~/NFDiffusion && git pull origin v2_local_implicit_unet && cd ASF && ./eval_sde.sh"
```

### Pull, Verify, and Report

```bash
ssh user@server << 'EOF'
cd ~/NFDiffusion
git pull origin v2_local_implicit_unet
echo "Files updated:"
git log -1 --stat
echo "Testing imports..."
cd ASF
python3 -c "from train_mamba_standalone import sde_sample; print('✓ Ready!')"
EOF
```

### Background Pull and Evaluation

```bash
ssh user@server << 'EOF'
cd ~/NFDiffusion
git pull origin v2_local_implicit_unet
cd ASF
nohup ./eval_sde.sh > eval_sde_output.log 2>&1 &
echo $! > eval_sde.pid
echo "Started in background, PID: $(cat eval_sde.pid)"
echo "Monitor with: tail -f ASF/eval_sde_output.log"
EOF
```

---

## 🔄 Update Workflow Summary

```bash
# 1. SSH to server
ssh user@your-server.com

# 2. Navigate to project
cd /path/to/NFDiffusion

# 3. Pull latest code
git pull origin v2_local_implicit_unet

# 4. Verify
ls -lh ASF/eval_sde*

# 5. Run evaluation
cd ASF
./eval_sde.sh

# 6. Check results
cat eval_sde_multiscale/metrics_comparison.txt
```

---

## 📝 Notes

- **Branch**: `v2_local_implicit_unet` (not main/master)
- **New Files**: 6 files added (2 scripts, 4 docs)
- **Dependencies**: Same as before (torch, numpy, matplotlib, scikit-image)
- **Checkpoints**: Existing checkpoints work without retraining
- **Backwards Compatible**: Old scripts still work

---

## ✅ Success Checklist

- [ ] Git pull completed without errors
- [ ] New files exist: `eval_sde.sh`, `eval_sde_multiscale.py`
- [ ] Scripts are executable: `chmod +x eval_sde.sh`
- [ ] Python can import: `sde_sample`, `ddim_sample`
- [ ] Help works: `python3 eval_sde_multiscale.py --help`
- [ ] Ready to run: `./eval_sde.sh`

---

## 🆘 Need Help?

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify you're on the correct branch: `git branch`
3. Check the commit: `git log -1 --oneline`
4. Review the README: `cat ASF/README_SDE.md`
5. Test minimal example: `python3 -c "from train_mamba_standalone import sde_sample"`

**Expected commit**: `11afab3` or later
**Expected files**: 6 new files in ASF/
