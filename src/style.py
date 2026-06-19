"""Shared publication style for the Dark Stack figures.

Restrained, ink-on-paper aesthetic suited to an MIT Press / Antikythera volume:
near-black ink, a muted categorical palette that survives greyscale printing,
thin rules, no chartjunk.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#1a1a1a"
MUTED = "#6e6e6e"
FAINT = "#b8b3a8"
PAPER = "#ffffff"

# Muted categorical palette (distinguishable in greyscale by lightness ordering)
PALETTE = {
    "frontier": "#2f6f8f",   # steel blue   -- mean-based frontier / U*
    "core":     "#9c3d2e",   # brick red    -- swap-regret core / V
    "neutral":  "#1a1a1a",   # ink
    "accent":   "#c08a2e",   # ochre
    "green":    "#4a7a4a",   # sage
    "violet":   "#6a5a8c",   # muted violet
    "warn":     "#a23b2e",
}
SEQ = ["#2f6f8f", "#9c3d2e", "#c08a2e", "#4a7a4a", "#6a5a8c", "#1a1a1a"]


def apply():
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "font.family": "serif",
        "font.serif": ["Palatino", "Palatino Linotype", "Book Antiqua",
                       "Georgia", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e7e3da",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "text.color": INK,
        "axes.labelcolor": INK,
    })


def panel_tag(ax, tag, dx=-0.02, dy=1.04):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="right", family="serif")


def fig_caption(fig, text, y=-0.02):
    fig.text(0.5, y, text, ha="center", va="top", fontsize=7.6, color=MUTED,
             family="serif", wrap=True)


def drop_legends_below_labels(fig, gap=0.06):
    """Post-layout pass: place every below-axes legend just beneath its own x-axis
    label, whatever the label's height. Robust to tall (subscript/fraction) labels
    that a fixed offset can't clear. Call after tight_layout, before savefig."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        leg = ax.get_legend()
        xlab = ax.xaxis.get_label()
        if leg is None or not xlab.get_text().strip():
            continue
        inv = ax.transAxes.inverted()
        lg = leg.get_window_extent(r).transformed(inv)
        if lg.y1 > 0.0:                       # only adjust legends that sit BELOW the axes
            continue
        xl = xlab.get_window_extent(r).transformed(inv)
        leg.set_bbox_to_anchor((0.5, xl.y0 - gap), transform=ax.transAxes)
    fig.canvas.draw()


def legend_below(ax, ncol=2, y=-0.34, fontsize=7.3, handles=None, labels=None):
    """Place a legend BELOW the axes (outside the data rectangle)."""
    kw = dict(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              fontsize=fontsize, frameon=False, handlelength=1.5,
              columnspacing=1.1, handletextpad=0.5, borderaxespad=0.0)
    if handles is not None:
        return ax.legend(handles, labels, **kw)
    return ax.legend(**kw)


def margin_text(fig, x, y, text, fontsize=7.0, color=None, ha="center", va="top"):
    """Text in the figure margin (outside any axes)."""
    return fig.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
                    color=color or MUTED, family="serif")
