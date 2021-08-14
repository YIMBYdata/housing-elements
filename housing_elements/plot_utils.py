from __future__ import annotations
import geopandas as gpd
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap

def register_cmap():
    if ('RedGreen' in plt.colormaps()):
        return
    cdict = {'red':  ((0.0, 0.0, 1.0),
                   (0.05, 1, 1.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 0.4, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.2, 0.8, 0.8),
                   (0.3, 0.9, .9),
                   (1.0, 0.8, 1)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.0, 0.0, 0.0),
                   (0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }

    cdict['alpha'] = ((0.0, .7, .7),
                   (0.25, 1, 1),
                   (0.75, 1.0, 1.0),
                   (1.0, 1.0, 1.0))

    cmap = LinearSegmentedColormap('RedGreen', cdict)
    plt.register_cmap('RedGreen', cmap)


def map_qoi(qoi, results_df):
    """ Save map for column name QOI in RESULTS_DF
    """
    results_copy = results_df.copy()
    results_copy['city'] = results_copy['City']
    results_copy['RHNA Success'] = results_copy['RHNA Success'] * 100
    bay = gpd.read_file('data/raw_data/bay_area_map/bay.shp')
    bay['city'] = bay['city'].str.title()
    bay['county'] = bay['county'].str.title()
    result = bay.merge(results_copy, how='inner', on='city')
    to_plot = result.to_crs(epsg=3857)
    qoi_in_title = qoi.title()
    legend_label = qoi
    file_name_prefix = qoi.lower()
    if qoi == 'RHNA Success':
        qoi_in_title = qoi
    if qoi == 'RHNA Success':
        legend_label = 'Percentage of RHNA Total Built'
    title = f'Map Of {qoi_in_title}'
    map_qoi_inner(qoi=qoi,
                  title=title,
                  legend_label=legend_label,
                  to_plot=to_plot,
                  file_name_prefix=file_name_prefix)


def map_qoi_inner(qoi, title, legend_label, to_plot, file_name_prefix):
    fig, ax = plt.subplots(figsize=(15, 15))
    register_cmap()
    plt.rcParams.update({'font.size': 25})
    to_plot.plot(ax=ax, column=qoi, legend=True,
                 legend_kwds={'label': legend_label, 'ax': ax}, cmap='RedGreen')

    plt.rcParams.update({'font.size': 10})
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    #ax.set_title(f' {title}', fontdict={'fontsize': 25})
    file_name_prefix = file_name_prefix.replace('/', '')
    file_name_prefix = file_name_prefix.replace(' ', '_')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, attribution=False)
    plt.savefig(f'figures/{file_name_prefix.lower()}_bay_map.jpg')
    fname = f'results/csvs_for_plots/{file_name_prefix.lower()}_bay_map.csv'
    if qoi == 'P(dev) for inventory':
        to_plot[['city', 'geometry', qoi]].to_csv(fname)


def catplot_qoi(result_df, qoi_col_prefix, order=None):
    assert 'City' in result_df.columns
    tiny_df = result_df.copy()
    relevant_qoi = [c for c in result_df.columns if qoi_col_prefix in c]
    tiny_df = tiny_df[relevant_qoi + ['City']]
    rename_map = {c: c[len(qoi_col_prefix) + 1:] for c in tiny_df.columns if c.startswith(qoi_col_prefix)}
    tiny_df.rename(rename_map, inplace=True, axis=1)
    long_df = pd.melt(tiny_df, id_vars='City', var_name='Method', value_name=qoi_col_prefix)
    sea.set(rc={'figure.figsize':(40,4)})
    ax = sea.barplot(x="City", y=qoi_col_prefix, hue="Method",
                data=long_df, saturation=.5, ci=None, order=order[:len(order)//3])
    ax.tick_params(axis='x', labelrotation=90)
    plt.savefig(f'figures/{qoi_col_prefix.lower()}_by_city_barplot.jpg')
