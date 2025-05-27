import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind


def barchart_1x2(df: pd.DataFrame, y_column: str, x_columns_list: list, order_bars: bool = False, xlabels_font_size: int = 8) -> None:
    """Plot barchart for two features against a single measure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for i, column in enumerate(x_columns_list):
        if order_bars:
            order = df.groupby(column)[y_column].mean().sort_values(ascending=False).index
            sns.barplot(x=column, y=y_column, data=df, ax=axes[i], order=order)
        else:
            sns.barplot(x=column, y=y_column, data=df, ax=axes[i])
        axes[i].set_xticks(axes[i].get_xticks())
        axes[i].set_xticklabels(axes[i].get_xticklabels(), fontsize=xlabels_font_size)
        axes[i].set_title(f"{y_column} distribution for {column}")
    plt.show()


def boxplot_1x2(df: pd.DataFrame, x_column: str, y_columns_list: list) -> None:
    """Plot boxplot for two features and a measure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for i, column in enumerate(y_columns_list):
        sns.boxplot(ax=axes[i], x=x_column, y=column, data=df)

        if x_column == "lateflight":
            # means, p-value
            on_time = df[df[x_column] == 0].dropna()[column]
            late = df[df[x_column] == 1].dropna()[column]
            t_stat, p_value = ttest_ind(on_time, late)
            axes[i].set_title(
                f"Box Plot of {column} by {x_column} \n (means: on-time({on_time.mean():.2f}) , late({late.mean():.2f})) \n (p-value: {p_value:.4f})"
            )
        else:
            axes[i].set_title(f"Box Plot of {column} by {x_column}")

        axes[i].set_xlabel(x_column)
        axes[i].set_ylabel(column)

    plt.show()


def corr_heatmap(df: pd.DataFrame, target_column: str | None = None, font_size: int = 12) -> None:
    """Calculate correlation coefficients and plot the heatmap."""
    correlation_matrix = df.select_dtypes(include=["number"]).corr()
    if target_column:
        correlation_matrix = correlation_matrix[[target_column]].drop(target_column)
        print(correlation_matrix.sort_values(by=target_column, ascending=False))
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": font_size})
    plt.title(f"{[target_column]} Correlation Matrix of Numerical Variables")
    plt.show()
