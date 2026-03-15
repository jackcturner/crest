import matplotlib.pyplot as plt
from pathlib import Path
import shutil

style_file = Path(__file__).parent.parent / "crest.mplstyle"
plt.style.use(style_file)

def _latex_available():
	candidates = ("pdflatex", "xelatex", "lualatex", "latex")
	for cmd in candidates:
		if shutil.which(cmd):
			return True
	return False

# If the style requests LaTeX but no LaTeX binary is on PATH, disable it.
if not _latex_available():
	print("Could not find LaTeX executable on PATH. Disabling LaTeX rendering in plots.")
	if plt.rcParams.get("text.usetex", False):
		plt.rcParams["text.usetex"] = False