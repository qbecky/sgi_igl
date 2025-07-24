import cv2
import ffmpeg
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import os
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial import ConvexHull

retval = cv2.VideoWriter_fourcc(*'mp4v')
    
def produce_video_from_path(
        im_pattern, save_folder, anim_fn, 
        framerate=25, transparent=False, overwrite_anim=False
    ):
    '''Concatenates images into one video
    
    Args:
        im_pattern: string representing the pattern of the images to be concatenated
        save_folder: string representing the folder where the video will be saved
        anim_fn: string representing the name of the video
        framerate: int representing the framerate of the video
        transparent: bool representing whether the images have transparency
        overwrite_anim: bool representing whether to overwrite the video if it already exists
    '''
    video_name = os.path.join(save_folder, anim_fn)
    if transparent:
        (
            ffmpeg
            .input(im_pattern, pattern_type='sequence', framerate=framerate)
            .filter('crop', w='trunc(iw/2)*2', h='trunc(ih/2)*2')
            .output(video_name, vcodec='png')
            .run(overwrite_output=overwrite_anim)
        )
    else:
        (
            ffmpeg
            .input(im_pattern, pattern_type='sequence', framerate=framerate)
            .filter('crop', w='trunc(iw/2)*2', h='trunc(ih/2)*2')
            .output(video_name, vcodec='libopenh264', pix_fmt='yuv420p')
            .run(overwrite_output=overwrite_anim)
        )

def interpolate_data(data, n_inter_frames, smooth=True):
    '''Interpolate data using cubic splines or linear interpolation
    
    Args:
        data: (n, d) array representing the data to be interpolated
        n_inter_frames: int representing the number of frames to be interpolated between each pair of knots
        smooth: bool representing whether to use cubic splines or linear interpolation
        
    Returns:
        data_interpolated: (n_inter_frames * (n - 1) + n, d) array representing the interpolated data
        ts: (n_inter_frames * (n - 1) + n,) array representing the time steps
        knots: (n,) array representing the knots
    '''

    n_knots = data.shape[0]
    knots = np.linspace(0.0, 1.0, n_knots)
    n_steps = n_inter_frames * (n_knots - 1) + n_knots
    ts = np.linspace(0.0, 1.0, n_steps)
    
    if smooth:
        interp = lambda x, y: CubicSpline(x, y, axis=0)
    else:
        interp = lambda x, y: interp1d(x, y, kind='linear', axis=0)

    spline_data = interp(knots, data)
    data_interpolated = spline_data(ts)
    return data_interpolated, ts, knots

def plot_violin_statistics(
    list_stats, list_names, colors=None, ylim=None, 
    title=None, xlabel=None, ylabel=None,
    filename=None, showText=True,
):
    '''
    Args:
        list_stats: a list of arrays giving the statistics to plot
        list_names: a list of names corresponding to each statistics
        colors: a list of colors for each deployments
        ylim: [yMin, yMax] used for all plots
        title: the title of the plot
        xlabel: the label of the x axis
        ylabel: the label of the y axis
        filename: the path to where we want to save the figure
        showText: whether we show text or not
    '''
    fig = plt.figure(figsize=(10, 8))

    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    parts = ax.violinplot(
        list_stats, showmeans=False, showmedians=False,
        showextrema=False
    )
    
    if colors is None:
        cmap = plt.get_cmap("Set2")
        colors = [cmap(i) for i in range(len(list_stats))]

    for c, pc in zip(colors, parts['bodies']):
        # pc.set_facecolor('#8797B2')
        pc.set_facecolor(c)
        # pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    perc5, quartile1, med, quartile3, perc95 = np.percentile(list_stats, [5, 25, 50, 75, 95], axis=1)
    inds = np.arange(1, len(quartile1) + 1)
    ax.scatter(inds, med, marker='o', color='white', s=90, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=13)
    ax.vlines(inds, perc5, perc95, color='k', linestyle='-', lw=3.5)

    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(list_names) + 1))
    if showText:
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticklabels(list_names)
    else:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
    ax.set_xlim(0.25, len(list_names) + 0.75)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    if not ylim is None: ax.set_ylim(ylim)

    if not filename is None: plt.savefig(filename)

    plt.show()


def plot_sdf_2d(xs, ys, sdfs, n_levels=20, show_text=True, filename=None):
    
    '''
    Args:
        xs (torch.tensor of shape (n_plots_x,)): x coordinates of the points on the grid
        ys (torch.tensor of shape (n_plots_y,)): y coordinates of the points on the grid
        sdfs (torch.tensor of shape (n_plots_x * n_plots_y,)): the corresponding sdfs to plot
        n_levels: the number of levels to show
        show_text: whether to show the text or not
        filename: the name of the file to save
    '''
    
    assert xs.shape[0] == sdfs.shape[0] and ys.shape[0] == sdfs.shape[1], "check the dimensions of xs, ys, and sdfs"
    
    min_sdf, max_sdf = torch.min(sdfs), torch.max(sdfs)
    max_abs_sdf = max(torch.abs(min_sdf), torch.abs(max_sdf))
    levels = np.linspace(-max_abs_sdf, max_abs_sdf, n_levels)
    
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.contourf(xs, ys, sdfs, levels=levels, cmap='coolwarm')
    ax.set_aspect('equal')
    if not show_text:
        ax.axis('off')
    if not filename is None: plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def checkerboard_circle_radial(ax, center=None, radius=1.0, n_rings=8, n_sectors=16, colors=('white', 'black'), lwidth=2):
    '''
    # Example usage to test the function
    fig, ax = plt.subplots(figsize=(6, 6))
    checkerboard_circle_radial(ax, radius=1.0, n_rings=8, n_sectors=16)
    plt.show()
    '''
    if center is None:
        center = (0, 0)
    
    for i in range(n_rings):
        r0 = radius * i / n_rings
        r1 = radius * (i + 1) / n_rings
        for j in range(n_sectors):
            theta0 = 2 * np.pi * j / n_sectors
            theta1 = 2 * np.pi * (j + 1) / n_sectors
            color = colors[(i + j) % 2]
            wedge = plt.Polygon([
                [center[0] + r0 * np.cos(theta0), center[1] + r0 * np.sin(theta0)],
                [center[0] + r1 * np.cos(theta0), center[1] + r1 * np.sin(theta0)],
                [center[0] + r1 * np.cos(theta1), center[1] + r1 * np.sin(theta1)],
                [center[0] + r0 * np.cos(theta1), center[1] + r0 * np.sin(theta1)]
            ], closed=True, color=color)
            ax.add_patch(wedge)
    theta = np.linspace(0, 2 * np.pi, n_sectors+1)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, color='k', linewidth=lwidth)
    ax.set_aspect('equal')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.axis('off')

def checkerboard_circle_aligned(
    ax, center=None, radius=1.0, n_rows=8, n_cols=8, 
    colors=('white', 'black'), lwidth=2, res=20, zorder=0,
):
    '''
    # Example usage to test the function
    fig, ax = plt.subplots(figsize=(6, 6))
    checkerboard_circle_aligned(ax, center=(0, 0), radius=1.0, n_rows=8, n_cols=8)
    plt.show()
    '''

    if center is None:
        center = (0, 0)
    x0, y0 = center
    size_x = 2 * radius / n_cols
    size_y = 2 * radius / n_rows
    for i in range(n_rows):
        for j in range(n_cols):
            # Rectangle corners
            x_min = x0 - radius + j * size_x
            x_max = x_min + size_x
            y_min = y0 - radius + i * size_y
            y_max = y_min + size_y
            # Vertices of the rectangle
            vertices = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max)
            ]
            # Create a grid for the rectangle
            xx = np.linspace(x_min, x_max, res)
            yy = np.linspace(y_min, y_max, res)
            grid = np.array(np.meshgrid(xx, yy)).reshape(2, -1).T
            # Mask points inside the circle
            mask = (grid[:, 0] - x0)**2 + (grid[:, 1] - y0)**2 < radius**2
            if np.any(mask):
                color = colors[(i + j) % 2]
                # Get the boundary of the intersection
                points_in_circle = grid[mask]
                # Convex hull for boundary (if enough points)
                if len(points_in_circle) >= 3:
                    hull = ConvexHull(points_in_circle)
                    poly_verts = points_in_circle[hull.vertices]
                    path = Path(poly_verts)
                    patch = PathPatch(path, facecolor=color, edgecolor=None, zorder=zorder)
                    ax.add_patch(patch)
                else:
                    # If only a few points, fallback to rectangle
                    rect = plt.Rectangle((x_min, y_min), size_x, size_y, facecolor=color, edgecolor=None, zorder=zorder)
                    ax.add_patch(rect)
    circ = Circle(center, radius, fill=False, edgecolor='k', linewidth=lwidth, zorder=zorder)
    ax.add_patch(circ)
    ax.set_aspect('equal')
    ax.set_xlim(x0 - radius, x0 + radius)
    ax.set_ylim(y0 - radius, y0 + radius)
    ax.axis('off')

def checkerboard_rectangle_aligned(
    ax, center=None, rect_width=1.5, rect_height=1.0, n_rows=8, n_cols=8, 
    colors=('white', 'black'), lwidth=2, zorder=0,
):
    '''
    # Example usage to test the function
    fig, ax = plt.subplots(figsize=(6, 6))
    checkerboard_rectangle_aligned(ax, center=(0, 0), rect_width=1.5, rect_height=1.0, n_rows=8, n_cols=8)
    plt.show()
    '''
    if center is None:
        center = (0, 0)
    x0, y0 = center
    size_x = rect_width / n_cols
    size_y = rect_height / n_rows
    rect_x_min = x0 - rect_width / 2
    rect_y_min = y0 - rect_height / 2

    for i in range(n_rows):
        for j in range(n_cols):
            x_min = rect_x_min + j * size_x
            y_min = rect_y_min + i * size_y
            color = colors[(i + j) % 2]
            rect = plt.Rectangle((x_min, y_min), size_x, size_y, facecolor=color, edgecolor=None, zorder=zorder)
            ax.add_patch(rect)
    # Draw the rectangle boundary
    rect_patch = plt.Rectangle(
        (rect_x_min, rect_y_min), rect_width, rect_height, fill=False, edgecolor='k', linewidth=lwidth, zorder=zorder+0.5
    )
    ax.add_patch(rect_patch)
    ax.set_aspect('equal')
    ax.set_xlim(rect_x_min, rect_x_min + rect_width)
    ax.set_ylim(rect_y_min, rect_y_min + rect_height)
    ax.axis('off')


