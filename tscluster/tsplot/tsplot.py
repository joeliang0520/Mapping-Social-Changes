from __future__ import annotations
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib
import mpl_toolkits
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D
import sys
import math
from tscluster.preprocessing.utils import broadcast_data

def _data_validator(
        X: npt.NDArray[np.float64]|None = None, 
        cluster_centers: npt.NDArray[np.float64]|None = None, 
        labels: npt.NDArray[np.float64]|None = None        
    ) -> None:

    """function to check the shapes of the data. Raises error if there are any inconsistenties in dimension of the data"""

    data = (X, cluster_centers, labels)
    valid_shapes = [{3}, {2, 3}, {2, 1}]
    names = ('X', 'cluster_centers', 'labels') 

    for i, j, k in zip(data, valid_shapes, names):
        if i is not None:
            if i.ndim not in j:
                raise TypeError(f"Invalid ndim. Expected {k}'s dimension to be any of {j} but got {i.ndim}")

    if cluster_centers is not None and labels is not None:
        if cluster_centers.shape[-2] != len(np.unique(labels)):
            raise ValueError(f"Number of clusters in cluster_centers and labels are not the same, they are {cluster_centers.shape[-2]} and {len(np.unique(labels))} respectively")
        elif cluster_centers.ndim == 3 and labels.ndim == 2 and cluster_centers.shape[0] != labels.shape[1]:
            raise ValueError(f"Number of timesteps in cluster_centers and labels are not the same, they are {cluster_centers.shape[0]} and {labels.shape[1]} respectively")

    if cluster_centers is not None and X is not None:
        if cluster_centers.shape[-1] != X.shape[-1]:
            raise ValueError(f"Number of features in cluster_centers and input data (X) are not the same, they are {cluster_centers.shape[-1]} and {X.shape[-1]} respectively")
        elif cluster_centers.ndim == 3 and cluster_centers.shape[0] != X.shape[0]:
            raise ValueError(f"Number of timesteps in cluster_centers and input data (X) are not the same, they are {cluster_centers.shape[0]} and {X.shape[0]} respectively")
     
    if labels is not None and X is not None:
        if labels.shape[0] != X.shape[1]:
            raise ValueError(f"Number of entities in labels and input data (X) are not the same, they are {labels.shape[0]} and {X.shape[1]} respectively")
        elif labels.ndim == 2 and labels.shape[1] != X.shape[0]:
            raise ValueError(f"Number of timesteps in labels and input data (X) are not the same, they are {labels.shape[1]} and {X.shape[0]} respectively")

def _get_shape(
        X: npt.NDArray[np.float64]|None = None, 
        cluster_centers: npt.NDArray[np.float64]|None = None, 
        labels: npt.NDArray[np.float64]|None = None 
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int]]:

    """
    Function to return the shape of the input data
    """

    if X is not None:
        X_shape = X.shape
    else:
        X_shape = (0, 0, 0)

    if cluster_centers is not None:
        if cluster_centers.ndim == 3:
            cc_shape = cluster_centers.shape 
        elif cluster_centers.ndim == 2:
            cc_shape = (0, *cluster_centers.shape)
    else:
        cc_shape = (0, 0, 0)

    if labels is not None:
        if labels.ndim == 2:
            l_shape = labels.shape 
        elif labels.ndim == 1:
            l_shape = (labels.shape[0], 0)
    else:
        l_shape = (0, 0)

    return X_shape, cc_shape, l_shape

def plot(
        *,
        X: npt.NDArray[np.float64]|None = None, 
        cluster_centers: npt.NDArray[np.float64]|None = None, 
        labels: npt.NDArray[np.float64]|None = None, 
        entity_idx: List[int]|None = None,
        entities_labels: List[str]|None = None,
        label_dict: dict|None = None,
        annot_fontsize: float|int = 10,
        show_all_entities: bool = True,
        figsize: Tuple[float, float] | None = None,
        shape_of_subplot: Tuple[int, int]|None = None, 
        xlabel: str|None = 'timesteps', 
        ylabel: str|None = 'value',
        cluster_labels: List[str]|None = None,
        title_list: List[str]|None = None,
        show_all_xticklabels: bool = True, 
        x_rotation: float|int = 45,
        show_X_marker: bool = False,
        show_cluster_center_marker: bool = False,
        ) -> Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]]:
    
    """
    Function to plot subplots of the time series data, cluster centers and label assignemnts. One subplot per feature. This is built on top of matplotlib.


    Parameters
    ----------
    X : numpy array, default=None
        The time series data in TNF format.
    cluster_centers : numpy array, default=None
        If numpy array, it is expected to be a 3D in TNF format. Here, N is the number of clusters. 
        If 2-D array, then it is interpreted as a K x F array where K is the number of clusters, and F is the number of features. Suitable for fixed cluster centers clustering.
    labels : numpy array, default=None
        It is expected to be a 2D array of shape (N, T) . Where N is the number of entities and T is the number of time steps. The value of the ith row at the t-th column is the label (cluster index) entity i was assigned to at time t.
        If 1-D array, it is interpreted as an array of length N. Where N is the number of entities. In such case, the i-th element is the cluster the i-th entity was assigned to across all time steps. Suitable for fixed assignment clustering.
    entity_idx : list, default=None 
        list of index of entities to display in the plot. If `show_all_entities` is True, `entity_idx` will be interpreted as the index of entities to be annonated.
    entities_labels : list, default=None
        list of labels for annotating the entities in `entity_idx`. If None, then labels of `entity_idx` in `label_dict` are used.
    label_dict dict, default=None
        a dictionary whose keys are 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
        - 'T' is a list of names/labels of each time step to be used as index of each dataframe. If None, range(0, T) is used. Where T is the number of time steps in the fitted data
        - 'N' is a list of names/labels of each entity. If None, range(0, N) is used. Where N is the number of entities/observations in the fitted data 
        - 'F' is a list of names/labels of each feature to be used as column of each dataframe. If None, range(0, F) is used. Where F is the number of features in the fitted data 
        If label_dict is not None, it is used to label timestep, entities, and features in the plot.   
    annot_fontsize : float|int, default=10
        The font size to be used for annotating entities in `entity_idx`.
    show_all_entities : bool, default=True
        If True, displays all the entities in `X` in the plot. If False, only entities in `entity_idx` are plotted.
    figsize : tuple, default=None
        The size of the figure. Should be tuple of length 2, where the first value is width of the figure, while the second value is the height of the figure.
    shape_of_subplot : tuple, default=None
        The shape of the subplots. Should be tuple of length 2, where the first value is number of rows, while the second value is the of columns in the figure.
        If None, (F, 1) is used. Where F is the number of features in `X` or `cluster_centers`.
    xlabel : str, default='timesteps'
        The label for the x-axis of each subplots
    ylabel : str, default='timesteps'
        The label for the y-axis of each subplots
    cluster_labels : list, default=None
        The labels to use for the clusters. If None, list(range(K)) is used. Where `K` is the number of clusters in `cluster_centers` or `labels`.
    title_list : list, default=None
        The title of each subplot (or feature).
    show_all_xticklabels : bool, default=True
        If True, shows labels of all time steps in the plot. If False, some may be suppressed (depending on the size of the plot)
    x_rotation : float or int, default=45
        The angle (in degrees) to rotate the timestep labels 
    show_X_marker : bool, default=False
        If True, show markers in the time series plot of X.
    show_cluster_center_marker : bool, default=False
        If True, show markers in the time series plot of the cluster centers.

    Returns
    -------
    matplotlib.figure.Figure
        the matplotlib figure object
    list
        a list of the matplotlib axes, one axes per subplot. 
    """
    
    _data_validator(X=X, cluster_centers=cluster_centers, labels=labels)

    X_shape, cc_shape, l_shape = _get_shape(X=X, cluster_centers=cluster_centers, labels=labels)

    # determine T. Should be the longest in X, cluster_centers and labels. this is to allow for variable number of input variables ie X, cluster_centers and labels
    T = max(X_shape[0], cc_shape[0], l_shape[1])
    N = max(X_shape[1], l_shape[0])
    F = max(X_shape[2], cc_shape[2])
    K = cc_shape[1]
    
    if labels is not None:
        K = max(K, len(np.unique(labels)))

    K = max(K, 1)

    label_dict_init = {'T': T, 'N': N, 'F': F}

    if label_dict is None:
        label_dict = {}
    
    for key, val in label_dict_init.items():
            _ = label_dict.setdefault(key, list(range(val)))

    cmap = pl.cm.get_cmap('rainbow')

    # broadcast cluster centers and labels if need be. This is done at this point because we need to compute T before now
    cluster_centers, labels = broadcast_data(T, cluster_centers=cluster_centers, labels=labels)

    norm = plt.Normalize(vmin=0, vmax=K-1)
    
    if cluster_labels is None:
        cluster_labels = list(map(str, range(K)))

    # determine the number of feature

    fig = plt.figure(figsize=figsize)

    if shape_of_subplot is None:
        shape_of_subplot = (F, 1)

    if title_list is None:
        title_list = ['Feature ' + str(f+1) if isinstance(f, int) else 'Feature ' + f for f in label_dict['F']]

    axes = []

    for f in range(F):
        ax = fig.add_subplot(*shape_of_subplot, f+1)

        if X is not None:
            if show_all_entities:
                idx = np.arange(X.shape[1])
            else:
                idx = entity_idx

            marker = ''
            if show_X_marker:
                marker = '.'

            # plot all data for feature f
            plt.plot(range(X.shape[0]), X[:, idx, f], c='k', ls='--', alpha=0.5, marker=marker)

            if entity_idx is not None:
                for li, i in enumerate(entity_idx):
                    if entities_labels is None:
                        e_labels = label_dict['N'][i]
                    else:
                        e_labels = entities_labels[li]

                    annot_i = np.random.choice(np.arange(len(X[:, i, f])), 1)[0]
                    annot_xy = list(enumerate(X[:, i, f]))[annot_i]

                    plt.annotate(e_labels, xy=annot_xy, xytext=(annot_xy[0]+0.5, annot_xy[1]+0.5), fontsize=annot_fontsize,
                                arrowprops=dict(facecolor='green',shrink=0))
                    
            if labels is not None:
            # scatter plot for marker for label assignment of data points. 
                for i in idx:
                    c = labels[i] 
                    plt.scatter(range(X.shape[0]), X[:, i, f], color=cmap(norm(c)), s=10)

                # label legend for cluster centers to match that of label assignment
                if cluster_centers is None:
                    for j in range(K):
                        plt.plot([], [], color=cmap(norm(j)), label=cluster_labels[j])  

        # plot of cluster centers
        if cluster_centers is not None:

            marker = ''
            if show_cluster_center_marker:
                marker = '.'

            for j in range(K):
                plt.plot(
                    range(cluster_centers.shape[0]), 
                    cluster_centers[:, j, f], 
                    color=cmap(norm(j)), 
                    label=cluster_labels[j],
                    marker=marker
                    )

        ax.set_xlabel(xlabel)

        ax.set_ylabel(ylabel)
        
        ax.set_title(title_list[f])
        
        ax.set_xticks(ticks=list(range(T))) # needed so that xticks wouldn't be float
        if show_all_xticklabels:
            ax.set_xticklabels(label_dict['T'], rotation=x_rotation)  

        if f != F-1:
            ax.set_xlabel('')
            ax.set_xticklabels('')
        
        if cluster_centers is not None or labels is not None:
            plt.legend()

        axes.append(ax)

    return fig, axes

def waterfall_plot(
        time_series: npt.NDArray[np.float64],
        label_dict: dict|None = None,
        *,
        xlabel: str = 'time-axis',
        ylabel: str = 'Features-axis',
        zlabel: str = 'Feature Values',
        title: str|None = None
        ) -> Tuple[matplotlib.figure.Figure, mpl_toolkits.mplot3d.axes3d.Axes3D]:
    
    """
    Function to plot a waterfall plot of a single time series data. This data can be a time series of a single entity or cluster center.
    To make the plot interactive, use a suitable matplotlib's magic command. E.g. `%matplotlib widget`. See this site for more: https://matplotlib.org/stable/users/explain/figure/interactive.html

    Parameters
    ----------
    time_series : numpy array
        The time series data to plot. This data can be a time series of a single entity or cluster center.
        Data should be a 2-D array of shape (T, F), where T and F are the number of timesteps and features respectively.
    label_dict : dict, default=None
        a dictionary whose keys are 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
        - 'T' is a list of names/labels of each time step to be used as index of each dataframe. If None, range(0, T) is used. Where T is the number of time steps in the fitted data
        - 'N' (ignored) is a list of names/labels of each entity. If None, range(0, N) is used. Where N is the number of entities/observations in the fitted data 
        - 'F' is a list of names/labels of each feature to be used as column of each dataframe. If None, range(0, F) is used. Where F is the number of features in the fitted data 
        If label_dict is not None, it is used to label timestep and features in the plot. 
    xlabel : str, default='time-axis'
        The label of the x-axis (time axis)
    ylabel : str, default='Features-axis'
        The label of the y-axis (feature axis)
    zlabel : str, default='Feature Values'
        The label of the z-axis (axis for the values of the features)
    title : str, default=None
        The title of the plot

    Returns
    -------
    matplotlib.figure.Figure
        the matplotlib figure object
    mpl_toolkits.mplot3d.axes3d.Axes3D
        the matplotlib's 3-D axes object. 
    """

    x = np.arange(time_series.shape[0]) # timesteps
    y = np.arange(time_series.shape[1]) # features
    
    X, Y = np.meshgrid(x, y)

    X, Y = X.T, Y.T

    # Creating a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotting the basic 3D surface
    ax.plot_surface(X, Y, time_series, color='grey', alpha=0.9)
    
    T, F = time_series.shape
    
    for f in range(F):
        ax.plot(x, [f]*T, time_series[:, f], color='red')
    
    label_dict_init = {'T': T, 'F': F}

    if label_dict is None:
        label_dict = {}
    
    for key, val in label_dict_init.items():
            _ = label_dict.setdefault(key, list(range(1, val+1)))

    x_tick_labels = label_dict['T']
    y_tick_labels = label_dict['F']

    if title is None:
        title = ''

    # Customizing the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.set_xticks(x, x_tick_labels)
    ax.set_yticks(y, y_tick_labels)
    
    # Create sliders for elevation and azimuth angles
    ax_elev = plt.axes([0.1, 0.05, 0.8, 0.02])
    ax_azim = plt.axes([0.1, 0.02, 0.8, 0.02])
    
    def _update(val):
        ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
        fig.canvas.draw_idle()
        
    slider_elev = Slider(ax_elev, 'Elevation', 0, 360, valinit=45)
    slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=45)
    
    # Connect sliders to update function
    slider_elev.on_changed(_update)
    slider_azim.on_changed(_update)
    
    # Displaying the plot
    return fig, ax

def plots(model,
        Xs = None,
        Zs = None,
        ys = None,
        show_clusters: None|List[int]=None, 
        show_cluster_center_only: bool=False,
        highlight_change: bool=False, 
        highlight_sample: None|str|int = None, 
        select_features: None|List[int] = None,
        show_annotations:bool = False,
        jitter: None|int= None,
        show_stats: bool = False,
        hide_others: bool = False,
        show_time_point: bool = False,
        **kwargs) -> plt.Figure:
    if Xs is None:
        Xs = model.X_
    T = Xs.shape[0]
    N = Xs.shape[1]
    if Zs is None:
        Zs = model.cluster_centers_
        #if Zs is two dimensional, then it is assumed to be a fixed cluster centers clustering, adding a time dimension
        if Zs.ndim == 2:
            Zs = np.array([Zs for _ in range(T)])
    n_clusters = Zs.shape[1]
    if ys is None:
        try:
            ys = model.Cs_hats_[-1].reshape((T, N, n_clusters))
        except AttributeError:
            ys = model._Cs.reshape((T, N, n_clusters))
    if select_features is not None:
        Xs = Xs[:,:,select_features]
        Zs = Zs[:,:,select_features]
        model.label_dict_['F'] = [model.label_dict_['F'][i] for i in select_features]
    width_per_plot = 7
    height_per_plot = 7
    #setting the gap between the time points in the x_ticks label to aovid overlapping
    gap = int(Xs.shape[0] // 8)
    if gap == 0:
        gap = 1

    # setting the subplot dimensions for multiple features dataset
    if Xs.shape[2] == 1:
        fig, axes = plt.subplots(1, 1, figsize=(width_per_plot, height_per_plot))
    
    elif Xs.shape[2]//3 == 0:
        fig, axes = plt.subplots(1, Xs.shape[2], figsize=(width_per_plot * Xs.shape[2], height_per_plot))
    elif Xs.shape[2]%3 > 0:
        fig, axes = plt.subplots(Xs.shape[2]//3 + 1, 3, 
                                figsize=(width_per_plot * 3, height_per_plot * ((Xs.shape[2]//3) +1)))
    else:
        fig, axes = plt.subplots(Xs.shape[2]//3, 3, 
                                figsize=(width_per_plot * 3, height_per_plot * (Xs.shape[2]//3)))
    
    # setting the all clusters to be plotted if 'show_clusters' not provided
    if show_clusters is None and ys is not None:
        show_clusters = np.arange(ys.shape[2])
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    if highlight_sample is not None and highlight_sample[0] in model.label_dict_['N']:
        highlight_sample = [np.where(np.array(model.label_dict_['N']) == i)[0][0] for i in highlight_sample]

    # setting the color palette for the clusters
    if 'color_palette' in kwargs:
        color_palette = kwargs['color_palette']
    else:
        color_palette = sns.color_palette('colorblind',n_clusters)
    
    if 'marker_palette' in kwargs:
        marker_palette = kwargs['marker_palette']
    else:
        marker_palette = ['x', 'o', 's', 'd', 'v', '^', '<', '>', 'p', 'h']

    for d in range(Xs.shape[2]):
        if not show_cluster_center_only:
            label_count = 0
            for n in range(Xs.shape[1]):
                label_change_flag = False
                if ys is not None:
                    labels = np.argmax(ys[:, n, :], axis=1)
                else:
                    labels = np.full(Xs.shape[0], -1)
                years = np.arange(0,Xs.shape[0],dtype=float)
                if jitter is not None:
                    for year in range(len(years)):
                        random_noise = np.random.uniform(-jitter,jitter)
                        years[year] += random_noise
                if ys is None or (any([l in show_clusters for l in labels])):
                    if len(set(labels)) > 1:
                        label_change_flag = True

                    if highlight_change and len(set(labels)) > 1:
                        for l in range(0, len(years) - 1):
                            if  labels[l] in show_clusters:
                                axes[d//3, d%3].plot([years[l], years[l+1]], [Xs[l, n, d], Xs[l+1, n, d]], 
                                            c = color_palette[labels[l]], lw = 2,ls='-', alpha=0.5)
                            else:
                                axes[d//3, d%3].plot([years[l], years[l+1]], [Xs[l, n, d], Xs[l+1, n, d]], 
                                            c = 'grey', lw = 2,ls='-', alpha=0.5)
                        axes[d//3, d%3].scatter(years, Xs[:, n, d], s = 100, alpha=0.5, 
                                        c = [color_palette[i] for i in labels])
                        
                        if show_annotations:
                            axes[d//3, d%3].annotate(model.label_dict_['N'][n], (Xs.shape[0] + gap, Xs[-1,n,d]), 
                                            textcoords="offset points", xytext=(0,0),ha='center')
                            width = (Xs[:,:,d].max().astype(int) - Xs[:,:,d].min().astype(int) )/ 100
                            axes[d//3, d%3].arrow(Xs.shape[0] + gap, Xs[-1,n,d], -0.8 - gap, 0,
                                        head_width= width, fc='blue', ec='red')
                    
                    elif highlight_sample is not None and n in highlight_sample:
                        line_styles = ['-', '-', '-']
                        if model.verbose and d == 0:
                            print(f'Label: {labels}', file=sys.stdout)
                        for l in range(0, len(years) - 1):
                            if  labels[l] in show_clusters:
                                axes[d//3, d%3].plot([years[l], years[l+1]], [Xs[l, n, d], Xs[l+1, n, d]], 
                                            c = color_palette[labels[l]], lw = 5,ls=line_styles[highlight_sample.index(n)])
                            else:
                                axes[d//3, d%3].plot([years[l], years[l+1]], [Xs[l, n, d], Xs[l+1, n, d]], 
                                            c = 'grey', lw = 5,ls='--')
                        if show_time_point:
                            for time in range(labels.shape[0]):
                                axes[d//3, d%3].scatter(years[time], Xs[time, n, d], s = 150, alpha=1, 
                                                        c = color_palette[labels[time]], marker=marker_palette[labels[time]])
                        if show_annotations and d != 0:
                            axes[d//3, d%3].annotate(model.label_dict_['N'][n], (Xs.shape[0] + gap, Xs[-10,n,d]), 
                                            textcoords="offset points", xytext=(0,0)
                                                                                ,ha='center', fontsize = 15, color = 'black')
                            width = (Xs[:,:,d].max().astype(int) - Xs[:,:,d].min().astype(int) )/ 100
                            
                    else:
                        if len(show_clusters) <n_clusters and highlight_sample is None and (not highlight_change):
                            for l in range(0, len(years) - 1):
                                    if labels[l] in show_clusters:
                                        axes[d//3, d%3].plot([years[l], years[l+1]], [Xs[l, n, d], Xs[l+1, n, d]], 
                                                    c = color_palette[labels[l]], lw = 2,ls='-')
                                    else:
                                        axes[d//3, d%3].plot([years[l], years[l+1]], [Xs[l, n, d], Xs[l+1, n, d]], 
                                                    c = 'grey', lw = 1,ls='--')
                            if show_time_point:
                                axes[d//3, d%3].scatter(years, Xs[:, n, d], s = 50, c = [color_palette[i] for i in labels], marker = marker_palette[labels[0]], alpha=0.5)
                        elif not hide_others and (not highlight_change) and (highlight_sample is None):
                            for l in range(0, len(years) - 1):
                                    axes[d//3, d%3].plot([years[l], years[l+1]], [Xs[l, n, d], Xs[l+1, n, d]], 
                                                c = color_palette[labels[l]], lw = 1,ls='--', alpha=0.5)
                            if show_time_point:
                                axes[d//3, d%3].scatter(years, Xs[:, n, d], s = 10, c = [color_palette[i] for i in labels],alpha=0.2)
                        elif not hide_others:
                            axes[d//3, d%3].plot(Xs[:, n, d], alpha=0.2, c = 'grey', ls='--')
                        if show_annotations and len(show_clusters) <n_clusters \
                                and (not highlight_change) and (highlight_sample is None):
                            included_entites = []
                            for n_n in range(Xs.shape[1]):
                                if (np.argmax(ys[:, n_n, :], axis=1)[0] in show_clusters) or (np.argmax(ys[:, n_n, :], axis=1)[-1] in show_clusters):
                                        included_entites.append(n_n)
                            axes[d//3, d%3].annotate(model.label_dict_['N'][n], (Xs.shape[0], Xs[-1,n,d]), 
                                            textcoords="axes fraction", xytext=(0.75, (np.sort(Xs[-1,included_entites,d], axis=None).tolist().index(Xs[-1,n,d]))/len(included_entites) + 0.05),
                                            arrowprops=dict(facecolor='black', shrink=0.05, width = 0.05, headwidth = 5,headlength = 5), fontsize = 12, alpha=0.8)
                            label_count += 20
                            width = (Xs[:,:,d].max().astype(int) - Xs[:,:,d].min().astype(int) )/ 100

                elif not hide_others:
                        axes[d//3, d%3].plot(Xs[:, n, d], alpha=0.2, c = 'grey', ls='--')

        if Zs is not None:
            for j in show_clusters:
                if highlight_change:
                    axes[d//3, d%3].plot(Zs[:, j, d], alpha=1, c = color_palette[j], ls='--', lw = 4)
                else:
                    axes[d//3, d%3].plot(Zs[:, j, d], alpha=1, c = color_palette[j], ls='--', lw = 4)
                    axes[d//3, d%3].scatter(0, Zs[0, j, d], c = color_palette[j], s = 100, alpha=1, marker=marker_palette[j])
                    axes[d//3, d%3].scatter(Xs.shape[0]-1, Zs[Xs.shape[0]-1, j, d], c = color_palette[j], s = 100, alpha=1, marker=marker_palette[j])
        if show_annotations:
            #maxmium length of characters in label
            max_len = max([len(str(i)) for i in model.label_dict_['N']])//3 
            if max_len < 3:
                max_len = 3
            axes[d//3, d%3].set_xticks(np.arange(0, Xs.shape[0] + max_len*gap, gap, dtype=int), 
                                        np.append(model.label_dict_['T'][::gap], ['']*max_len))
        else:
            axes[d//3, d%3].set_xticks(np.arange(0, Xs.shape[0], gap), model.label_dict_['T'][::gap])
        
        if d % 3 == 0:
            axes[d//3, d%3].set_ylabel('Feature')
        if d == len(model._label_dict_['F']) - 1:
            legend_elements = []

            if 'cluster_name' in kwargs:
                #sort the cluster name and return the sorted index
                sorted_index = np.argsort(kwargs['cluster_name'])
                for i in sorted_index:
                    if i in show_clusters:
                        legend_elements += [Line2D([0], [0], color=color_palette[i], ls= '--',lw=2, marker=marker_palette[i], markersize=7,
                                                label='Center of '+ kwargs['cluster_name'][i])]
                if len(show_clusters) <n_clusters:
                    legend_elements += [Line2D([0], [0], color=color_palette[i], ls= '-',lw=2,
                                            label='Entity in '+ kwargs['cluster_name'][i]) for i in show_clusters]
            else:   
                legend_elements += [Line2D([0], [0], color=color_palette[i], ls= '--',lw=3, marker=marker_palette[i], markersize=7,
                                            label=f'Definition of Cluster {i + 1}') for i in show_clusters]
                if len(show_clusters) <n_clusters:
                    legend_elements += [Line2D([0], [0], color=color_palette[i], ls= '-',lw=3,
                                            label=f'Time Points in Cluster {i + 1}') for i in show_clusters]
                    
            if not show_cluster_center_only and not hide_others:
                if highlight_sample is not None:
                    legend_elements += [Line2D([0], [0], color='grey', ls= '--',lw=1, label='Other Entities')]
                elif label_change_flag or show_clusters != list(range(model.n_clusters)):
                    legend_elements += [Line2D([0], [0], color='grey', ls= '--',lw=1, label='Entities in other Clusters')]                         
            if 'bbox_to_anchor' in kwargs:
                leg = axes[d//3, d%3].legend(handles=legend_elements, loc='best',fontsize = 14, bbox_to_anchor=kwargs['bbox_to_anchor'])
                leg.get_frame().set_linewidth(0.0)
            else:
                leg = axes[d//3, d%3].legend(handles=legend_elements, loc='best',fontsize = 12,frameon=False)
        
            
        y_gap = (math.ceil(Xs[:,:,d].max()) - math.floor(Xs[:,:,d].min())) / 10
        axes[d//3, d%3].set_yticks(np.arange(math.floor(Xs[:,:,d].min()) , math.ceil(Xs[:,:,d].max()) + y_gap, y_gap))
        axes[d//3, d%3].set_ylim(Xs[:,:,d].min() - y_gap, Xs[:,:,d].max() + y_gap)
        axes[d//3, d%3].set_title(model.label_dict_['F'][d])

        if show_stats:
            axes[d//3, d%3].plot(np.median(Xs[:,:,d], axis=1), c = 'black', ls='-.', lw = 5, alpha=0.1)
            axes[d//3, d%3].plot(np.quantile(Xs[:,:,d], 0.25, axis=1), c = 'black', ls='-.', lw = 5,alpha=0.1)
            axes[d//3, d%3].plot(np.quantile(Xs[:,:,d], 0.75, axis=1), c = 'black', ls='-.', lw = 5,alpha=0.1)
            axes[d//3, d%3].plot(np.min(Zs[:,:,d], axis=1), c = 'black', ls='-.', lw = 5,alpha=0.1)
            axes[d//3, d%3].plot(np.max(Zs[:,:,d], axis=1), c = 'black', ls='-.', lw = 5,alpha=0.1)
            axes[d//3, d%3].annotate('Median', (0, np.median(Xs[:,:,d], axis=1)[0]))
            axes[d//3, d%3].annotate('Q1', (0, np.quantile(Xs[:,:,d], 0.25, axis=1)[0]))
            axes[d//3, d%3].annotate('Q3', (0, np.quantile(Xs[:,:,d], 0.75, axis=1)[0]))

    return fig, axes