import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ── Data (edit these values) ──────────────────────────────────────────────────
runs_data = {
    # method        gflops_per_sample  dice    context_size  input_size
    'downsample':  {'method': 'Downsample to 128',  'gflops_per_sample': 13.57,  'dice': 0.2482, 'context_size': 1,  'input_size': 128},
    'original':  {'method': 'Original size',  'gflops_per_sample': 54.29,  'dice': 0.3032, 'context_size': 1,  'input_size': 128},
    'hierarchical':  {'method': 'Hierarchical',  'gflops_per_sample': 27.14,  'dice': 0.1601, 'context_size': 3,  'input_size': 128},
    #'crop':  {'method': 'Hierarchical - Crop to target GT',  'gflops_per_sample': 27.14,  'dice': 0.2571, 'context_size': 8,  'input_size': 128},
    'Ours':  {'method': 'Ours',  'gflops_per_sample': 21.03,  'dice': 0.3341, 'context_size': 8,  'input_size': 128},
}
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.title_fontsize': 8,
    'figure.autolayout': True,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
})

df = pd.DataFrame.from_dict(runs_data, orient='index')
dice_col = 'dice'


def plot_pareto(df, vary_col, fixed_col, fixed_val, title, filename):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    plot_df = df[df[fixed_col] == fixed_val].dropna(subset=['gflops_per_sample', dice_col])
    if plot_df.empty:
        print(f"No data for {fixed_col}={fixed_val}")
        return

    color_map = {
        'Downsample to 128':     'tab:green',
        'Original size': 'tab:orange',
        'Hierarchical':        'tab:red',
        'Hierarchical - Crop to target GT':      '#f4a582',
        'Ours':                'tab:blue',
    }
    methods = plot_df['method'].unique()

    for method in methods:
        mdf = plot_df[plot_df['method'] == method]
        ax.scatter(
            mdf['gflops_per_sample'], mdf[dice_col],
            color=color_map[method],
            label=method, alpha=0.9, zorder=3, s=40,
        )
        for _, row in mdf.iterrows():
            ax.annotate(
                method,
                (row['gflops_per_sample'], row[dice_col]),
                textcoords="offset points", xytext=(0, 8),
                ha='center', va='bottom', fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc=color_map[method], ec='none', alpha=0.15),
            )

    ax.set_xlabel('Computational Cost (GFLOPs / Sample)')
    ax.set_ylabel('Validation Dice')
    ax.set_title(title, pad=6)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.margins(x=0.25, y=0.30)

    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


plot_pareto(
    df,
    vary_col='context_size',
    fixed_col='input_size',
    fixed_val=128,
    title='Computational cost vs. accuracy',#\n Original size vs downsampled input',
    filename='pareto.pdf',
)
