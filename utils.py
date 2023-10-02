"""Defines auxiliary functions for fixing the seeds, setting
a logger and visualizing WeatherBench data."""
import logging
import os
import random

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import AxesGrid


# Tensorboard visualization titles.
TITLES = ("Upsampled with interpolation",
          "Super-resolution reconstruction",
          "High-resolution original")


def set_seeds(seed: int = 0):
    """Sets random seeds of Python, NumPy and PyTorch.
    Args:
        seed: Seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def dict2str(dict_obj: dict, indent_l: int = 4) -> str:
    """Converts dictionary to string for printing out.
    Args:
        dict_obj: Dictionary or OrderedDict.
        indent_l: Left indentation level.
    Returns:
        Returns string version of opt.
    """
    msg = ""
    for k, v in dict_obj.items():
        if isinstance(v, dict):
            msg = f"{msg}{' '*(indent_l*2)}{k}:[\n{dict2str(v, indent_l+1)}{' '*(indent_l*2)}]\n"
        else:
            msg = f"{msg}{' '*(indent_l*2)}{k}: {v}\n"
    return msg


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    """Sets up the logger.
    Args:
        logger_name: The logger name.
        root: The directory of logger.
        phase: Either train or val.
        level: The level of logging.
        screen: If True then write logging records to a stream.
    """
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    log_file = os.path.join(root, "{}.log".format(phase))
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)


def construct_and_save_wbd_plot(latitude: np.array, longitude: np.array, single_variable: torch.tensor,
                                path: str, title: str = None, label: str = None, dpi: int = 200,
                                figsize: tuple = (11, 8.5), cmap: str = "coolwarm", vmin=None,
                                vmax=None, costline_color="black"):
    """Creates and saves WeatherBench data visualization for a single variable.
    Args:
        latitude: An array of latitudes.
        longitude: An array of longitudes.
        single_variable: A tensor to visualize.
        path: Path of a directory to save visualization.
        title: Title of the figure.
        label: Label of the colorbar.
        dpi: Resolution of the figure.
        figsize: Tuple of (width, height) in inches.
        cmap: A matplotlib colormap.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        costline_color: Matplotlib color.
    """
    # single_variable, longitude = add_cyclic_point(single_variable, coord=np.array(longitude))
    plt.figure(dpi=dpi, figsize=figsize)
    projection = ccrs.PlateCarree()
    ax = plt.axes()#projection=projection)

    if cmap == "binary":
        # For mask visualization.
        p = plt.contourf(longitude, latitude, single_variable, 10, transform=projection,
                         cmap=(matplotlib.colors.ListedColormap(["white", "gray", "black"])
                               .with_extremes(over="0.25", under="0.75")),
                         vmin=-2, vmax=2)
        boundaries, ticks = [-1, -0.33, 0.33, 1], [-1, 0, 1]
    elif cmap == "coolwarm":
        # For temperature visualization.
        p = plt.contourf(longitude, latitude, single_variable, 60, transform=projection, cmap=cmap,
                         levels=np.linspace(vmin, vmax, max(int(np.abs(vmax-vmin))//2, 3)))
        boundaries, ticks = None, np.round(np.linspace(vmin, vmax, 7), 2)

    elif cmap == "Greens":
        # For visualization of standard deviation.
        p = plt.contourf(longitude, latitude, single_variable, 60, transform=projection, cmap=cmap,
                         extend="max")
        boundaries, ticks = None, np.linspace(single_variable.min(), single_variable.max(), 5)

    # ax.set_xticks(np.linspace(-180, 180, 5), crs=projection)
    # ax.set_yticks(np.linspace(-90, 90, 5), crs=projection)
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # lat_formatter = LatitudeFormatter()
    # ax.xaxis.set_major_formatter(lon_formatter)
    # ax.yaxis.set_major_formatter(lat_formatter)
    # ax.coastlines(color=costline_color)

    # plt.colorbar(p, pad=0.06, label=label, orientation="horizontal", shrink=0.75,
    #              boundaries=boundaries, ticks=ticks)

    plt.title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close("all")


def add_batch_index(path: str, index: int):
    """Adds the number of batch gotten from data loader to path.
    Args:
        path: The path to which the function needs to add batch index.
        index: The batch index.
    Returns:
        The path with the index appended to the filename.
    """
    try:
        filename, extension = path.split(".")
    except ValueError:
        splitted_parts = path.split(".")
        filename, extension = ".".join(splitted_parts[:-1]), splitted_parts[-1]
    return f"{filename}_{index}.{extension}"


def construct_and_save_wbd_plots(latitude: np.array, longitude: np.array, data: torch.tensor,
                                 path: str, title: str = None, label: str = None,
                                 dpi: int = 200, figsize: tuple = (11, 8.5), cmap: str = "coolwarm",
                                 vmin=None, vmax=None, costline_color="black"):
    """Creates and saves WeatherBench data visualization.
    Args:
        latitude: An array of latitudes.
        longitude: An array of longitudes.
        data: A batch of variables to visualize.
        path: Path of a directory to save visualization.
        title: Title of the figure.
        label: Label of the colorbar.
        dpi: Resolution of the figure.
        figsize: Tuple of (width, height) in inches.
        cmap: A matplotlib colormap.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        costline_color: Matplotlib color.
    """
    if len(data.shape) > 2:
        data = data.squeeze()

    if len(data.shape) > 2:
        for batch_index in range(data.shape[0]):
            path_for_sample = add_batch_index(path, batch_index)
            construct_and_save_wbd_plot(latitude, longitude, data[batch_index], path_for_sample,
                                        title, label, dpi, figsize, cmap, vmin, vmax, costline_color)
    else:
        construct_and_save_wbd_plot(latitude, longitude, data, path, title, label, dpi, figsize, cmap,
                                    vmin, vmax, costline_color)


def construct_tb_visualization(latitude: np.array, longitude: np.array, data: tuple, label=None,
                               dpi: int = 300, figsize: tuple = (22, 6), cmap: str = "coolwarm") -> Figure:
    """Construct tensorboard visualization figure.
    Args:
        latitude: An array of latitudes.
        longitude: An array of longitudes.
        data: A batch of variables to visualize.
        label: Label of the colorbar.
        dpi: Resolution of the figure.
        figsize: Tuple of (width, height) in inches.
        cmap: A matplotlib colormap.
    Returns:
        Matplotlib Figure.
    """
    max_value = max((tensor.max() for tensor in data))
    min_value = min((tensor.min() for tensor in data))
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axgr = AxesGrid(fig, 111, axes_class=axes_class, nrows_ncols=(1, 3), axes_pad=0.95, cbar_location="bottom",
                    cbar_mode="single", cbar_pad=0.01, cbar_size="2%", label_mode='')
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()

    for i, ax in enumerate(axgr):
        single_variable, lon = add_cyclic_point(data[i], coord=np.array(longitude))
        ax.set_title(TITLES[i])
        ax.gridlines(draw_labels=True, xformatter=lon_formatter, yformatter=lat_formatter,
                     xlocs=np.linspace(-180, 180, 5), ylocs=np.linspace(-90, 90, 5))
        p = ax.contourf(lon, latitude, single_variable, transform=projection, cmap=cmap,
                        vmin=min_value, vmax=max_value)
        ax.coastlines()

    axgr.cbar_axes[0].colorbar(p, pad=0.01, label=label, shrink=0.95)
    fig.tight_layout()
    plt.close("all")
    return fig


def accumulate_statistics(new_info: dict, storage: dict):
    """Accumulates statistics provided with new_info into storage.
    Args:
        new_info: A dictionary containing new information.
        storage: A dictionary where to accumulate new information.
    """
    for key, value in new_info.items():
        if key in storage:
            storage[key].append(value)
        else:
            storage[key] = [value]


# def get_transformation(name: str) -> Transform:
#     """Return data transformation class corresponding to name.
#     Args:
#         name: The name of transformation.
#     Returns:
#         A data transformer.
#     """
#     if name == "LocalStandardScaling":
#         from weatherbench_data.transforms import LocalStandardScaling as Transformation
#     elif name == "GlobalStandardScaling":
#         from weatherbench_data.transforms import GlobalStandardScaling as Transformation
#     return Transformation


def get_optimizer(name: str) :
    """Return optimization algorithm class corresponding to name.
    Args:
        name: The name of optimizer.
    Returns:
        A torch optimizer.
    """
    if name == "adam":
        from torch.optim import Adam as Optimizer
    elif name == "adamw":
        from torch.optim import AdamW as Optimizer
    return Optimizer


def construct_mask(x: torch.tensor) -> torch.tensor:
    """Constructs signum(x) tensor with tolerance around 0 specified
    by torch.isclose function.
    Args:
        x: The input tensor.
    Returns:
        Signum(x) with slight tolerance around 0.
    """
    values = torch.ones_like(x)
    zero_mask = torch.isclose(x, torch.zeros_like(x))
    neg_mask = x < 0
    values[neg_mask] = -1
    values[zero_mask] = 0
    return values
def normal(y_pred):
    y_pred_max=torch.max(y_pred,-1,keepdim=True).values.max(-2,keepdim=True).values
    y_pred_min=torch.min(y_pred,-1,keepdim=True).values.min(-2,keepdim=True).values
    return (y_pred-y_pred_min)/(y_pred_max-y_pred_min+1e-16)

def psnr(y_pred,y_true):
    y_pred=normal(y_pred)
    y_true=normal(y_true)
    loss=10. * torch.log10(255. / torch.mean((y_pred - y_true) ** 2))
    return loss



# def my_plot(self,data_list):
#     for channel in range(5):
#         figure,axs=plt.subplots(5,5,figsize=(16,20))
#             for i in range(5):
#                 axs[0,i].imshow(pred[i,channel,:,:],vmin=-2,vmax=2,cmap='RdBu_r')
#                 axs[1,i].imshow(true[i,channel,:,:],vmin=-2,vmax=2,cmap='RdBu_r')
#                 axs[2,i].imshow(mask[i,0,:,:],vmin=-2,vmax=2,cmap='RdBu_r')
#                 axs[3,i].imshow(land[i,0,:,:],vmin=-2,vmax=2,cmap='RdBu_r')
#                 axs[4,i].imshow(test[i,channel,:,:],vmin=-2,vmax=2,cmap='RdBu_r')
#                 axs[0,i].set_title("RMSE:"+str(np.sqrt(np.mean((pred[i,channel,:,:]-true[i,channel,:,:])**2))))
#                 axs[0,0].set_ylabel("y_pred")
#             plt.savefig(self.configs_train.draw_path[channel]+"/epoch{}.png".format(epoch))
#             plt.close()
#     return fig



# def reverse_transform_candidates(candidates: torch.tensor, reverse_transform: Transform,
#                                  transformations: dict, variables: list, data_type: str,
#                                  months: list, tranform_monthly: bool):
#     """Reverse transforms.
#     Args:
#         candidates: A tensor of shape [n, B, C, H, W].
#         reverse_transform: A reverse transformation.
#         transformations: A dictionary of transformations.
#         variables: Weatherbench data variables.
#         data_type: Either 'lr' or 'hr'.
#         months: A list of months for each batch sample of length (B, ).
#         tranform_monthly: Either to apply transformation month-wise or not.
#     Returns:
#         Reversed transformed candidates.
#     """
#     for i in range(candidates.shape[0]):
#         candidates[i] = reverse_transform(candidates[i], transformations, variables,
#                                           data_type, months, tranform_monthly)