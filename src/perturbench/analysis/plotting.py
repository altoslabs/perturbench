import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text


def scatter_labels(
    x,
    y,
    df=None,
    hue=None,
    labels=None,
    label_size=6,
    x_title=None,
    y_title=None,
    axis_title_size=14,
    axis_text_size=12,
    plot_title=None,
    title_size=15,
    ax=None,
    figsize=None,
    hide_legend=True,
    size=60,
    alpha=0.8,
    xlim=None,
    ylim=None,
    ident_line=False,
    quadrants=False,
    **kwargs,
):
    """Generate a scatterplot with optional text labels for data points"""

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if df is not None:
        sns.scatterplot(ax=ax, x=x, y=y, hue=hue, data=df, marker=".", s=size, **kwargs)
    else:
        if type(size) in [int, float]:
            size = [size] * len(x)

        sns.scatterplot(ax=ax, x=x, y=y, hue=hue, marker=".", s=size, **kwargs)

    ax.set_xlabel(x_title, size=axis_title_size)
    ax.set_ylabel(y_title, size=axis_title_size)
    ax.tick_params(axis="x", labelsize=axis_text_size)
    ax.tick_params(axis="y", labelsize=axis_text_size)
    ax.set_title(plot_title, size=title_size)

    if xlim is not None:
        ax.set_xlim(xmin=xlim[0], xmax=xlim[1])
    if ylim is not None:
        ax.set_ylim(ymin=ylim[0], ymax=ylim[1])

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    min_pt = max(xlim[0], ylim[0])
    max_pt = min(xlim[1], ylim[1])
    if ident_line:
        ax.plot((min_pt, max_pt), (min_pt, max_pt), ls="--", color="red", alpha=0.7)

    if quadrants:
        ax.axvline(ls="--", color="red", alpha=0.3)
        ax.axhline(ls="--", color="red", alpha=0.3)

    if hide_legend and (ax.get_legend() is not None):
        ax.get_legend().remove()

    if labels is not None:
        assert df is not None
        texts = []
        for lab in labels:
            texts.append(ax.text(df.loc[lab, x], df.loc[lab, y], lab, size=label_size))

        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black"), ax=ax)


## Base boxplot
def boxplot_jitter(
    x,
    y,
    df,
    hue=None,
    x_title=None,
    y_title=None,
    axis_title_size=14,
    axis_text_size=12,
    jitter_size=10,
    alpha=0.8,
    figsize=None,
    plot_title=None,
    title_size=15,
    ax=None,
    violin=False,
):
    """Generate a boxplot or violin plot with jitter for the data points"""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if violin:
        sns.violinplot(
            ax=ax, x=x, y=y, hue=hue, data=df, palette=sns.color_palette("colorblind")
        )
        plt.setp(ax.collections, alpha=alpha)

    else:
        sns.boxplot(
            ax=ax,
            x=x,
            y=y,
            hue=hue,
            data=df,
            showfliers=False,
            palette=sns.color_palette("colorblind"),
        )
        for patch in ax.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, alpha))

    sns.stripplot(
        ax=ax,
        x=x,
        y=y,
        hue=hue,
        data=df,
        color="black",
        marker=".",
        size=jitter_size,
        dodge=True,
        label=False,
        legend=None,
    )

    ax.set_xlabel(x_title, size=axis_title_size)
    ax.set_ylabel(y_title, size=axis_title_size)
    ax.set_title(plot_title, size=title_size)
    ax.tick_params(axis="x", labelrotation=90, labelsize=axis_text_size)
    ax.tick_params(axis="y", labelsize=axis_text_size)

    if ax is None:
        plt.show()
