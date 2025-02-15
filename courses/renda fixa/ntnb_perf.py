from utils import Performance
from data import trackers_ntnb

df = trackers_ntnb()

perf = Performance(df)

print(perf.table)
perf.table.to_clipboard()

# perf.plot_drawdowns("NTNB 25y", show_chart=True)
# perf.plot_underwater("NTNB 25y", show_chart=True)
