"""
Layout auditor: enforce 'no text/legend inside the plotting area' and 'no
text/legend collides with data', by three independent methods.

METHOD 1 (inside-axes test): every Text and Legend must lie OUTSIDE the axes data
rectangle (0,1)x(0,1) in axes-fraction coordinates (titles/labels/panel-tags live
outside by construction; heatmap cell labels are exempt as tabular data).

METHOD 2 (data-collision test): no plotted data vertex (lines, scatter, bars) may
fall inside any Text/Legend bounding box (display pixels).

METHOD 3 is visual (done separately by eye on the PNGs).
"""
import os, sys, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle

HERE = os.path.dirname(__file__)
FIGS = ["fig1_opacity", "fig2_stackelberg", "fig3_versioning",
        "fig4_pathologies", "fig5_anything_factory", "robustness",
        "dynamics_invariance", "analytic_anchor", "opacity_dividend", "dividend_depth"]


def load_and_run(modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, modname + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.main()
    return plt.gcf()


def bbox_inside(inner, outer, tol=0.04):
    """Is `inner` bbox substantially within `outer` (axes-fraction) rectangle?"""
    # overlap area fraction of inner that is inside (tol,1-tol)
    x0 = max(inner.x0, tol); x1 = min(inner.x1, 1 - tol)
    y0 = max(inner.y0, tol); y1 = min(inner.y1, 1 - tol)
    if x1 <= x0 or y1 <= y0:
        return False
    inside = (x1 - x0) * (y1 - y0)
    area = (inner.x1 - inner.x0) * (inner.y1 - inner.y0)
    return area > 0 and inside / area > 0.5


def audit_figure(fig):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    violations = []
    for ai, ax in enumerate(fig.axes):
        is_heatmap = len(ax.images) > 0
        label = ax.get_title() or f"axes#{ai}"
        # --- collect text artists: ax.texts + legend ---
        texts = list(ax.texts)
        leg = ax.get_legend()
        text_boxes_disp = []
        for t in texts:
            if not t.get_text().strip():
                continue
            if is_heatmap:
                continue  # tabular cell labels are data, exempt
            disp = t.get_window_extent(rend)
            frac = disp.transformed(ax.transAxes.inverted())
            text_boxes_disp.append((f"text '{t.get_text()[:24]}'", disp))
            if bbox_inside(frac, frac.__class__.from_bounds(0, 0, 1, 1)):
                violations.append(f"[{label}] M1 INSIDE-AXES: text '{t.get_text()[:30]}' "
                                  f"at axes-frac x[{frac.x0:.2f},{frac.x1:.2f}] y[{frac.y0:.2f},{frac.y1:.2f}]")
        if leg is not None:
            disp = leg.get_window_extent(rend)
            frac = disp.transformed(ax.transAxes.inverted())
            text_boxes_disp.append(("legend", disp))
            if bbox_inside(frac, frac.__class__.from_bounds(0, 0, 1, 1)):
                violations.append(f"[{label}] M1 INSIDE-AXES: legend "
                                  f"at axes-frac x[{frac.x0:.2f},{frac.x1:.2f}] y[{frac.y0:.2f},{frac.y1:.2f}]")
        # --- METHOD 2: data vertices inside any text/legend box ---
        verts_disp = []
        for line in ax.get_lines():
            xy = line.get_xydata()
            if len(xy):
                verts_disp.append(ax.transData.transform(xy))
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                off = coll.get_offsets()
                if len(off):
                    verts_disp.append(ax.transData.transform(np.asarray(off)))
        for patch in ax.patches:
            if isinstance(patch, Rectangle) and patch.get_width() > 0:
                bb = patch.get_window_extent(rend)
                verts_disp.append(np.array([[bb.x0, (bb.y0+bb.y1)/2], [bb.x1, (bb.y0+bb.y1)/2],
                                            [(bb.x0+bb.x1)/2, bb.y1]]))
        allv = np.vstack(verts_disp) if verts_disp else np.empty((0, 2))
        # keep only VISIBLE data (inside the axes display rectangle); clipped data
        # whose transData coords fall outside the axes is not a real collision.
        axbb = ax.get_window_extent(rend)
        if len(allv):
            vis = ((allv[:, 0] >= axbb.x0 - 1) & (allv[:, 0] <= axbb.x1 + 1) &
                   (allv[:, 1] >= axbb.y0 - 1) & (allv[:, 1] <= axbb.y1 + 1))
            allv = allv[vis]
        for name, box in text_boxes_disp:
            if len(allv) == 0:
                continue
            inside = ((allv[:, 0] >= box.x0) & (allv[:, 0] <= box.x1) &
                      (allv[:, 1] >= box.y0) & (allv[:, 1] <= box.y1))
            if inside.any():
                violations.append(f"[{label}] M2 DATA-COLLISION: {name} overlaps "
                                  f"{int(inside.sum())} data vertices")
    return violations


def audit_strict_pixels(fig):
    """METHOD 3 (independent): flag ANY pixel overlap between a text/legend's
    rendered bounding box and the axes DATA rectangle (>2px), not just a >50%
    center test. Catches partial intrusions across the axes boundary."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    viol = []
    for ai, ax in enumerate(fig.axes):
        if len(ax.images) > 0:               # heatmap cell labels are tabular data
            continue
        ab = ax.get_window_extent(rend)
        items = [(f"text '{t.get_text()[:22]}'", t) for t in ax.texts if t.get_text().strip()]
        leg = ax.get_legend()
        if leg is not None:
            items.append(("legend", leg))
        title = ax.get_title() or f"axes#{ai}"
        for name, art in items:
            b = art.get_window_extent(rend)
            ox = min(ab.x1, b.x1) - max(ab.x0, b.x0)
            oy = min(ab.y1, b.y1) - max(ab.y0, b.y0)
            if ox > 2 and oy > 2:            # genuine 2-D overlap with data rectangle
                frac = (ox * oy) / max((b.x1 - b.x0) * (b.y1 - b.y0), 1)
                viol.append(f"[{title[:40]}] M3 PIXEL-OVERLAP: {name} intrudes "
                            f"{ox:.0f}x{oy:.0f}px into axes ({frac*100:.0f}% of its box)")
    return viol


def audit_text_text(fig):
    """METHOD 4: no text artist may overlap another (legend vs axis label, etc.) —
    the margin-collision class the inside-axes tests miss."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    viol = []
    for ai, ax in enumerate(fig.axes):
        items = []
        leg = ax.get_legend()
        if leg is not None:
            items.append(("legend", leg.get_window_extent(rend)))
        for nm, art in [("xlabel", ax.xaxis.get_label()), ("ylabel", ax.yaxis.get_label()),
                        ("title", ax.title)]:
            if art.get_text().strip():
                items.append((nm, art.get_window_extent(rend)))
        for t in ax.texts:
            if t.get_text().strip():
                items.append((f"text '{t.get_text()[:16]}'", t.get_window_extent(rend)))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (na, a), (nb, b) = items[i], items[j]
                ox = min(a.x1, b.x1) - max(a.x0, b.x0)
                oy = min(a.y1, b.y1) - max(a.y0, b.y0)
                if ox > 2 and oy > 2:
                    viol.append(f"[{(ax.get_title() or 'axes#'+str(ai))[:34]}] M4 TEXT-TEXT: "
                                f"{na} overlaps {nb} ({ox:.0f}x{oy:.0f}px)")
    return viol


def main():
    total = 0
    subset = [a for a in sys.argv[1:] if not a.startswith("-")]
    figs = [f for f in FIGS if any(s in f for s in subset)] if subset else FIGS
    for f in figs:
        fig = load_and_run(f)
        v = audit_figure(fig) + audit_strict_pixels(fig) + audit_text_text(fig)
        plt.close(fig)
        status = "OK (clean)" if not v else f"{len(v)} VIOLATION(S)"
        print(f"\n=== {f}: {status} ===")
        for line in v:
            print("   " + line)
        total += len(v)
    print(f"\n==== TOTAL VIOLATIONS: {total} ====")


if __name__ == "__main__":
    main()
