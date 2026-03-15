import matplotlib.pyplot as plt
from pathlib import Path

style_file = Path(__file__).parent.parent / "crest.mplstyle"
plt.style.use(style_file)