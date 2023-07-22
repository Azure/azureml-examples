import pandas as pd

from typing import Any, Dict, List, Optional, Tuple, Union
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _format_grain_name(grain: Union[str, Tuple[str], List[str]]) -> str:
    """
    Convert grain name to string.

    :param grain: the grain name.
    :return: the string representation of the given grain.
    """
    if not isinstance(grain, tuple) and not isinstance(grain, list):
        return str(grain)
    grain = list(map(str, grain))
    return "|".join(grain)


def draw_one_plot(
    df: pd.DataFrame,
    time_column_name: str,
    target_column_name: str,
    grain_column_names: List[str],
    columns_to_plot: List[str],
    pdf: PdfPages,
    plot_predictions=False,
) -> None:
    """
    Draw the single plot.

    :param df: The data frame with the data to build plot.
    :param time_column_name: The name of a time column.
    :param grain_column_names: The name of grain columns.
    :param pdf: The pdf backend used to render the plot.
    """
    if isinstance(grain_column_names, str):
        grain_column_names = [grain_column_names]
    fig, _ = plt.subplots(figsize=(20, 10))
    df = df.set_index(time_column_name)
    plt.plot(df[columns_to_plot])
    plt.xticks(rotation=45)
    if grain_column_names:
        grain_name = [df[grain].iloc[0] for grain in grain_column_names]
        plt.title(f"Time series ID: {_format_grain_name(grain_name)}")
    plt.legend(columns_to_plot)
    plt.close(fig)
    pdf.savefig(fig)


if __name__ == "__main__":
    pass
