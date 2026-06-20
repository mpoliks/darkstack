"""
Capstone: one Dark Factory, five views.

Instantiates a single DarkFactory, prints its five headline lenses, then renders
all five figures from that one object. This is the whole artifact in one command:

    python run_factory.py
"""
import os, sys

ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "figures"))

from darkfactory import DarkFactory
import fig1_opacity, fig2_stackelberg, fig3_versioning, fig4_pathologies, fig5_anything_factory


def main():
    f = DarkFactory(seed=2).run()
    print("=" * 64)
    print("ONE dark factory, five lenses")
    print("=" * 64)
    print("  opacity      ", f.lens_opacity())
    print("  stackelberg  ", f.lens_stackelberg())
    print("  versions     ", f.lens_versions())
    print("  pathologies  ", f.lens_pathologies())
    print("  entrainment  ", DarkFactory.lens_entrainment())
    print("\nRendering the five figures from the one object ...\n")
    for mod in (fig1_opacity, fig2_stackelberg, fig3_versioning,
                fig4_pathologies, fig5_anything_factory):
        mod.main()
    print("\nDone — figures in figures/, numbers in out/.")


if __name__ == "__main__":
    main()
