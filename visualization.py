"""
This file contains simple helper functions to support visualization code in analysis.ipynb
"""

import pandas as pd


def update_axes(fig, values):
    """
    Sets titles for x and y axes on plotly graphs.

    :param fig: plotly.graph_objects.Figure. Contains graphs for which axes titles are required to be set.
    :param values: dict. {'xtitle': str, 'ytitle': str, 'range': list, 'row': int, 'col': int}
    """
    for val in values:
        # Update xaxis and yaxis properties
        fig.update_xaxes(title_text=val['xtitle'], row=val['row'], col=val['col'], range=val['range'])
        fig.update_yaxes(title_text=val['ytitle'], row=val['row'], col=val['col'])


def count_positives(data, feature, offer, n_bins):
    """
    Counts percentages of positive responses given by customers for each bin.

    :param data: DataFrame. The data with customers' information.
    :param feature: str. Name of the feature to split into bins.
    :param offer: str. Name of the offer for which to count percentages of positive responses.
    :param n_bins: int. Number of bins to split into.
    :return: DataFrame. Contains percentage of positive responses for each bin.
    """
    if feature != 'gender':
        received = data[data[offer] != 0][[feature, offer]]     # consider only cases when the offer was received (!= 0)
        bins = pd.cut(received[feature], n_bins, precision=0)
        bins = pd.DataFrame({feature: bins, offer: received[offer]})
    else:   # splitting into bins is not required for 'gender' feature
        bins = data[data[offer] != 0][[feature, offer]]

    positive = bins[bins[offer] == 3].groupby(feature).count()
    total = bins.groupby(feature).count()
    percentage = (positive / total) * 100

    return percentage
