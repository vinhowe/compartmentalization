"""Conspicuous placeholder panels for figure slots that another panel will fill.

Each placeholder has exactly the geometry the real panel must have. To fill a
slot, draw the panel with the named FIGSIZE/ADJUST constants from
plot_baseline_val_curves.py (setup_paper_style(); fig.subplots_adjust(**ADJUST);
plain fig.savefig, no tight_layout or bbox_inches="tight") and replace the
PLACEHOLDER_*.pdf include in main.tex.

Usage: python make_figure_placeholders.py OUT_DIR
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from plot_baseline_val_curves import (placeholder_panel, GRID2_FIGSIZE, GRID2_ADJUST,
                                      THREEUP_FIGSIZE, THREEUP_ADJUST)

SLOTS = [
    ("PLACEHOLDER_fig4_panel_d_2x2.pdf", GRID2_FIGSIZE, GRID2_ADJUST,
     "PLACEHOLDER: Fig. 4 (d)\nGRID2_FIGSIZE 3.3x1.75in\nGRID2_ADJUST"),
    ("PLACEHOLDER_figwd_panel_c_3up.pdf", THREEUP_FIGSIZE, THREEUP_ADJUST,
     "PLACEHOLDER:\nweight decay (c)\nTHREEUP_FIGSIZE\n2.4x2.0in"),
]

out_dir = Path(sys.argv[1])
for name, figsize, adjust, text in SLOTS:
    placeholder_panel(out_dir / name, figsize, adjust, text=text, color="tab:red")
