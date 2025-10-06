#!/bin/bash

# Setup script to enable custom metrics in VERL by overriding metric_utils.py
# This script should be run before training to enable automatic custom metric detection
#
# NOTE: This script modifies files in the verl submodule, but these changes are NOT
# tracked by git because .gitmodules has "ignore = dirty" set for the verl submodule.
# This means:
#   - Changes to verl/ won't show in git status
#   - You need to run this script after cloning the repository
#   - The modifications are local only and won't be committed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCOUT_DIR="$(dirname "$SCRIPT_DIR")"
VERL_METRICS_FILE="$SCOUT_DIR/verl/verl/trainer/ppo/metric_utils.py"
SCOUT_METRICS_FILE="$SCOUT_DIR/scout/utils/metric_utils.py"

echo "======================================================================"
echo "Setting up custom metrics for Scout PPO training"
echo "======================================================================"

# Check if VERL metrics file exists
if [ ! -f "$VERL_METRICS_FILE" ]; then
    echo "Error: VERL metrics file not found at $VERL_METRICS_FILE"
    exit 1
fi

# Check if Scout custom metrics file exists
if [ ! -f "$SCOUT_METRICS_FILE" ]; then
    echo "Error: Scout custom metrics file not found at $SCOUT_METRICS_FILE"
    exit 1
fi

# Backup the original VERL metrics file if not already backed up
if [ ! -f "$VERL_METRICS_FILE.original" ]; then
    echo "Backing up original VERL metric_utils.py..."
    cp "$VERL_METRICS_FILE" "$VERL_METRICS_FILE.original"
    echo "  ✓ Backup created at $VERL_METRICS_FILE.original"
else
    echo "  ℹ Original backup already exists"
fi

# Copy Scout's custom metrics file to override VERL's version
echo "Installing Scout custom metrics..."
cp "$SCOUT_METRICS_FILE" "$VERL_METRICS_FILE"
echo "  ✓ Custom metrics installed"

echo ""
echo "======================================================================"
echo "Custom metrics setup complete!"
echo "======================================================================"
echo ""
echo "The following enhancements are now active:"
echo "  • Automatic detection of custom metrics from reward functions"
echo "  • Statistics (mean/max/min/rate/std) for all numeric metrics"
echo "  • Metrics will appear with 'custom/' prefix in logs"
echo ""
echo "To restore the original VERL metrics, run:"
echo "  cp $VERL_METRICS_FILE.original $VERL_METRICS_FILE"
echo ""
echo "You can now run training with automatic custom metric tracking:"
echo "  python scout/trainer/main_ppo.py"
echo "======================================================================"