# %%
import copy
import glob
import itertools
import os
import shutil
import sys
import warnings
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import ot
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LogNorm
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr
from sklearn import manifold
from sqlalchemy import URL, create_engine
from sqlalchemy_utils import create_database, database_exists, drop_database
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


class VisualizationConfig:
    """This is an instance for sharing the parameters to make the figures of GWOT with the instance PairwiseAnalysis.

    Please check the tutoial.ipynb for detailed info.
    """

    def __init__(
        self,
        show_figure: bool = True,
        fig_ext:str='png',
        figsize: Tuple[int, int] = (8, 6),
        cbar_label_size: int = 15,
        cbar_ticks_size: int = 10,
        cbar_format: Optional[str]=None,
        cbar_label: Optional[str]=None,
        xticks_size: int = 10,
        yticks_size: int = 10,
        xticks_rotation: int = 90,
        yticks_rotation: int = 0,
        tick_format: str = '%.2f',
        title_size: int = 20,
        legend_size: int = 5,
        xlabel: Optional[str] = None,
        xlabel_size: int = 15,
        ylabel: Optional[str] = None,
        ylabel_size: int = 15,
        zlabel: Optional[str] = None,
        zlabel_size: int = 15,
        color_labels: Optional[List[str]] = None,
        color_hue: Optional[str] = None,
        colorbar_label: Optional[str] = None,
        colorbar_range: List[float] = [0., 1.],
        colorbar_shrink: float = 1.,
        markers_list: Optional[List[str]] = None,
        marker_size: int = 30,
        alpha: int = 1,
        color: str = 'C0',
        cmap: str = 'cividis',
        ot_object_tick: bool = False,
        ot_category_tick: bool = False,
        draw_category_line: bool = False,
        category_line_color: str = 'C2',
        category_line_alpha: float = 0.2,
        category_line_style: str = 'dashed',
        plot_eps_log: bool = False,
        lim_eps: Optional[float] = None,
        lim_gwd: Optional[float] = None,
        lim_acc: Optional[float] = None,
    ) -> None:
        """Initializes the VisualizationConfig class with specified visualization parameters.

        Args:
            show_figure (bool, optional):
                Whether to display the figure. Defaults to True.
            fig_ext (str, optional):
                The extension of the figure. Defaults to 'png'.
            figsize (Tuple[int, int], optional):
                Size of the figure. Defaults to (8, 6).
            cbar_label_size (int, optional):
                Size of the colorbar label. Defaults to 15.
            cbar_ticks_size (int, optional):
                Size of the colorbar ticks. Defaults to 10.
            cbar_format (Optional[str]):
                Format of the colorbar. Defaults to None.
            cbar_label (Optional[str]):
                Title of the colorbar. Defaults to None.
            xticks_size (int, optional):
                Size of the xticks. Defaults to 10.
            yticks_size (int, optional):
                Size of the yticks. Defaults to 10.
            xticks_rotation (int, optional):
                Rotation angle of the xticks. Defaults to 90.
            yticks_rotation (int, optional):
                Rotation angle of the yticks. Defaults to 0.
            tick_format (Optional[str]):
                Format of the ticks. Defaults to '%.2f'.
            title_size (int, optional):
                Size of the title. Defaults to 20.
            legend_size (int, optional):
                Size of the legend. Defaults to 5.
            xlabel (str, optional):
                Label of the x-axis. Defaults to None.
            xlabel_size (int, optional):
                Size of the x-axis label. Defaults to 15.
            ylabel (str, optional):
                Label of the y-axis. Defaults to None.
            ylabel_size (int, optional):
                Size of the y-axis label. Defaults to 15.
            zlabel (str, optional):
                Label of the z-axis. Defaults to None.
            zlabel_size (int, optional):
                Size of the z-axis label. Defaults to 15.
            color_labels (List[str], optional):
                Labels of the color. Defaults to None.
            color_hue (str, optional):
                Hue of the color. Defaults to None.
            colorbar_label (str, optional):
                Label of the colorbar. Defaults to None.
            colorbar_range (list, optional):
                Range of the colorbar. Defaults to [0, 1].
            colorbar_shrink (float, optional):
                Shrink of the colorbar. Defaults to 1.
            markers_list (List[str], optional):
                List of markers. Defaults to None.
            marker_size (int, optional):
                Size of the marker. Defaults to 30.
            alpha (int, optional):
                Alpha of the marker. Defaults to 1.
            color (str, optional):
                Color for plots. Defaults to 'C0'.
            cmap (str, optional):
                Colormap of the figure. Defaults to 'cividis'.
            ot_object_tick (bool, optional):
                Whether to tick object. Defaults to False.
            ot_category_tick (bool, optional):
                Whether to tick category. Defaults to False.
            draw_category_line (bool, optional):
                Whether to draw category lines. Defaults to False.
            category_line_color (str, optional):
                Color for category lines. Defaults to 'C2'.
            category_line_alpha (float, optional):
                Alpha for category lines. Defaults to 0.2.
            category_line_style (str, optional):
                Style for category lines. Defaults to 'dashed'.
            plot_eps_log (bool, optional):
                Whether to plot in logarithmic scale for epsilon. Defaults to False.
            lim_eps (float, optional):
                Limits for epsilon. Defaults to None.
            lim_gwd (float, optional):
                Limits for GWD. Defaults to None.
            lim_acc (float, optional):
                Limits for accuracy. Defaults to None.
        """

        self.visualization_params = {
            'show_figure':show_figure,
            'fig_ext':fig_ext,
            'figsize': figsize,
            'cbar_label_size': cbar_label_size,
            'cbar_ticks_size': cbar_ticks_size,
            'cbar_format':cbar_format,
            'cbar_label':cbar_label,
            'xticks_size': xticks_size,
            'yticks_size': yticks_size,
            'xticks_rotation': xticks_rotation,
            'yticks_rotation': yticks_rotation,
            'tick_format': tick_format,
            'title_size': title_size,
            'legend_size': legend_size,
            'xlabel': xlabel,
            'xlabel_size': xlabel_size,
            'ylabel': ylabel,
            'ylabel_size': ylabel_size,
            'zlabel': zlabel,
            'zlabel_size': zlabel_size,
            'color_labels': color_labels,
            'color_hue': color_hue,
            'colorbar_label': colorbar_label,
            'colorbar_range': colorbar_range,
            'colorbar_shrink': colorbar_shrink,
            'alpha': alpha,
            'markers_list': markers_list,
            'marker_size': marker_size,
            'color':color,
            'cmap':cmap,
            'ot_object_tick': ot_object_tick,
            'ot_category_tick': ot_category_tick,
            'draw_category_line': draw_category_line,
            'category_line_color': category_line_color,
            'category_line_alpha': category_line_alpha,
            'category_line_style': category_line_style,
            'plot_eps_log':plot_eps_log,
            'lim_eps':lim_eps,
            'lim_ged':lim_gwd,
            'lim_acc':lim_acc,
        }

    def __call__(self) -> Dict[str, Any]:
        """Returns the visualization parameters.

        Returns:
            Dict[str, Any]: Dictionary containing the visualization parameters.
        """
        return self.visualization_params

    def set_params(self, **kwargs) -> None:
        """Allows updating the visualization parameters.

        Args:
            **kwargs: keyword arguments representing the parameters to be updated.
        """

        for key, item in kwargs.items():
            self.visualization_params[key] = item
