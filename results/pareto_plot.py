import matplotlib.pyplot as plt
import pandas as pd

# ── Data (edit these values) ──────────────────────────────────────────────────
runs_data = {
    # method        gflops_per_sample  dice    context_size  input_size
    'uni_ctx1':  {'method': 'universeg',  'gflops_per_sample': 1.2,  'dice': 0.61, 'context_size': 1,  'input_size': 128},
    'uni_ctx3':  {'method': 'universeg',  'gflops_per_sample': 3.4,  'dice': 0.67, 'context_size': 3,  'input_size': 128},
    'uni_ctx8':  {'method': 'universeg',  'gflops_per_sample': 8.1,  'dice': 0.71, 'context_size': 8,  'input_size': 128},
    'icl_ctx1':  {'method': 'patch_icl',  'gflops_per_sample': 0.9,  'dice': 0.63, 'context_size': 1,  'input_size': 128},
    'icl_ctx3':  {'method': 'patch_icl',  'gflops_per_sample': 2.5,  'dice': 0.70, 'context_size': 3,  'input_size': 128},
    'icl_ctx8':  {'method': 'patch_icl',  'gflops_per_sample': 6.4,  'dice': 0.75, 'context_size': 8,  'input_size': 128},
}
# ─────────────────────────────────────────────────────────────────────────────

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

    colors  = {'universeg': 'tab:orange', 'patch_icl': 'tab:blue'}
    markers = {'universeg': 'o',          'patch_icl': 's'}

    for method in plot_df['method'].unique():
        mdf = plot_df[plot_df['method'] == method].sort_values('gflops_per_sample')
        ax.plot(
            mdf['gflops_per_sample'], mdf[dice_col],
            marker=markers.get(method, 'o'),
            color=colors.get(method, 'k'),
            label=method, alpha=0.85,
        )
        for _, row in mdf.iterrows():
            ax.annotate(
                f"$n$={int(row[vary_col])}",
                (row['gflops_per_sample'], row[dice_col]),
                textcoords="offset points", xytext=(0, 6),
                ha='center', fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.75, lw=0.5),
            )

    ax.set_xlabel('Computational Cost (GFLOPs / Sample)')
    ax.set_ylabel('Validation Dice')
    ax.set_title(title, pad=6)
    ax.legend(title="Method", framealpha=0.85, edgecolor='gray')
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.margins(x=0.12, y=0.18)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


plot_pareto(
    df,
    vary_col='context_size',
    fixed_col='input_size',
    fixed_val=128,
    title='Computational cost vs. accuracy\n(Varying Context Size, Input: 128)',
    filename='pareto_context_size.pdf',
)
