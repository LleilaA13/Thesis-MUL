#!/bin/bash
# Repository Cleanup Script
# Run this to remove redundant files and reorganize the repository

set -e

echo "Starting repository cleanup..."

# Create backup directory
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating backup in $BACKUP_DIR..."

# Function to safely remove with backup
safe_remove() {
    if [ -e "$1" ]; then
        echo "Removing: $1"
        mv "$1" "$BACKUP_DIR/" 2>/dev/null || echo "  (already moved or doesn't exist)"
    fi
}

# 1. Remove test and debug scripts
echo ""
echo "=== Removing test and debug files ==="
safe_remove "test_device_fix.py"
safe_remove "test_unlearned_model.py"
safe_remove "colab_ready_code.py"

# 2. Remove old unlearning scripts (keeping src/classification for main code)
echo ""
echo "=== Removing standalone unlearning scripts ==="
safe_remove "generate_all_masks.py"
safe_remove "inspect_mask.py"
safe_remove "unlearn_config.py"
safe_remove "results_tracker.py"
safe_remove "unlearn_results_log.json"

# 3. Remove temporary visualization files
echo ""
echo "=== Removing temporary files ==="
safe_remove "output.png"
safe_remove "outputu.png"

# 4. Remove duplicate notebook at root
echo ""
echo "=== Removing duplicate notebooks ==="
safe_remove "activation_grids.ipynb"

# 5. Remove entire redundant directories
echo ""
echo "=== Removing redundant directories ==="
safe_remove "dummy_scripts"
safe_remove "archive"

# 6. Clean up .pt files if not needed (commented out - review first)
echo ""
echo "=== Model checkpoint files (review before removing) ==="
echo "The following .pt files are in root - consider moving to models/ or experiments/:"
ls -lh *.pt 2>/dev/null || echo "No .pt files in root"

# 7. Organize scripts directory
echo ""
echo "=== Organizing scripts directory ==="
if [ -d "scripts" ]; then
    echo "Scripts directory exists. Consider organizing by function:"
    echo "  - training/"
    echo "  - unlearning/"
    echo "  - analysis/"
    echo "  - visualization/"
fi

# 8. Clean Python cache
echo ""
echo "=== Cleaning Python cache ==="
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "=== Cleanup Summary ==="
echo "Backup created in: $BACKUP_DIR"
echo "Review the backup directory before deleting it"
echo ""
echo "Next steps:"
echo "1. Review the backup directory"
echo "2. Test that everything still works"
echo "3. If satisfied, delete the backup: rm -rf $BACKUP_DIR"
echo "4. Update git: git add -A && git commit -m 'Clean up repository structure'"
echo ""
echo "Cleanup complete!"
