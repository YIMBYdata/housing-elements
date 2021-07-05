from __future__ import annotations
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sea
from scipy import stats
import matplotlib.pyplot as plt
from housing_elements import utils, los_altos_permits, san_francisco_permits, san_jose_permits, map_utils
from pathlib import Path
import warnings
import os, sys
import pickle
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import statsmodels

# Silence an annoying warning that I get when running pd.read_excel
warnings.filterwarnings("ignore", message="Data Validation extension is not supported and will be removed")


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def load_sites_and_permits():
    sites_df = utils.load_all_sites()
    all_cities = sites_df.jurisdict.unique()

    assert len(all_cities) == 108

    cities_with_sites = {}
    for city in all_cities:
        with HiddenPrints():
            try:
                sites = utils.load_site_inventory(city)
                assert sites.shape[0]
                cities_with_sites[city] = sites
            except Exception:
                print("Loading sites failed for " + city, file=sys.stderr)

    assert len(cities_with_sites) == 106

    cities_with_permits = {}
    for city in all_cities:
        with HiddenPrints():
            try:
                cities_with_permits[city] = utils.load_all_new_building_permits(city)
            except Exception:
                print(city, file=sys.stderr)

    assert len(cities_with_permits) == 99
    assert len(set(cities_with_permits).intersection(set(cities_with_sites))) == 97

    return cities_with_sites, cities_with_permits

def get_results_for_city_kwargs(kwargs):
    with HiddenPrints():
        return get_results_for_city(**kwargs)

def parallel_process(function, args_list, num_workers=8):
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(function, args_list), total=len(args_list)))
    return results

def get_results_for_city(
    city: str,
    sites: gpd.GeoDataFrame,
    permits: gpd.GeoDataFrame,
    match_by: str,
    geo_matching_lax: bool = False,
) -> pd.DataFrame:
    nonvacant_matches, nonvacant_sites, nonvacant_ratio = utils.calculate_pdev_for_nonvacant_sites(
        sites, permits, match_by, geo_matching_lax
    )
    vacant_matches, vacant_sites, vacant_ratio = utils.calculate_pdev_for_vacant_sites(
        sites, permits, match_by, geo_matching_lax
    )
    all_matches, all_sites, all_ratio = utils.calculate_pdev_for_inventory(
        sites, permits, match_by, geo_matching_lax
    )

    return {
        'City': city,
        'Mean underproduction': utils.calculate_underproduction_on_sites(sites, permits),
        'RHNA Success': utils.calculate_rhna_success(city, permits),
        'P(inventory) for homes built': utils.calculate_pinventory_for_dev(sites, permits),
        'P(dev) for nonvacant sites': nonvacant_ratio,
        'P(dev) for vacant sites': vacant_ratio,
        'P(dev) for inventory': all_ratio,
        '# nonvacant matches': f'{nonvacant_matches} / {nonvacant_sites}',
        '# vacant matches': f'{vacant_matches} / {vacant_sites}',
        '# matches': f'{all_matches} / {all_sites}',
    }


def get_ground_truth_results_for_city(city: str) -> pd.DataFrame:
    if city == 'San Jose':
        permits = san_jose_permits.load_all_permits()
    elif city == 'San Francisco':
        permits = san_francisco_permits.load_all_permits()
    elif city == 'Los Altos':
        permits = los_altos_permits.load_all_permits()
    else:
        raise ValueError(f"Ground truth data not available for {city}")

    permits = utils.load_all_new_building_permits(city)
    sites = utils.load_site_inventory(city)

    if city == 'San Jose':
        # the San Jose data already has "San Jose, CA" at the end
        address_suffix = None
    else:
        address_suffix = ', ' + city + ', CA'

    if not isinstance(permits, gpd.GeoDataFrame):
        permits = gpd.GeoDataFrame(permits, geometry=None, crs='EPSG:3857')
    permits = utils.impute_missing_geometries(permits, address_suffix)

    return {
        'City': city,
        'Mean underproduction': utils.calculate_underproduction_on_sites(sites, permits),
        'RHNA Success': utils.calculate_rhna_success(city, permits),
        'P(inventory) for homes built': utils.calculate_pinventory_for_dev(sites, permits),
        'P(dev) for nonvacant sites': utils.calculate_pdev_for_nonvacant_sites(sites, permits, match_by='both', geo_matching_lax=True),
        'P(dev) for vacant sites': utils.calculate_pdev_for_vacant_sites(sites, permits, match_by='both', geo_matching_lax=True),
        'P(dev) for inventory': utils.calculate_pdev_for_inventory(sites, permits, match_by='both', geo_matching_lax=True),
    }


def get_additional_stats(results_both_df: pd.DataFrame):
    match_cols = {
        'All': '# matches',
        'Nonvacant': '# nonvacant matches',
        'Vacant': '# vacant matches',
    }

    p_dev_cols = {
        'All': 'P(dev) for inventory',
        'Nonvacant': 'P(dev) for nonvacant sites',
        'Vacant': 'P(dev) for vacant sites',
    }

    results = []
    for site_type in ['All', 'Nonvacant', 'Vacant']:
        sites_matches_col = results_both_df[match_cols[site_type]]
        num_matches = sites_matches_col.str.split('/').apply(lambda x: int(x[0]))
        num_sites = sites_matches_col.str.split('/').apply(lambda x: int(x[1]))

        p_dev_col = results_both_df[p_dev_cols[site_type]]

        results.append(
            {
                'Site type': site_type,
                'Overall development rate': '{:.1%}'.format(num_matches.sum() / num_sites.sum()),
                'Num sites': num_sites.sum(),
                'Median P(dev)': '{:.1%}'.format(8 / 5 * p_dev_col.median()),
                'Mean P(dev)': '{:.1%}'.format(8 / 5 * p_dev_col.mean()),
            }
        )

    return pd.DataFrame(results)

def make_plots(results_both_df: pd.DataFrame) -> None:
    utils.map_qoi('P(dev) for inventory', results_both_df)
    utils.map_qoi('P(dev) for vacant sites', results_both_df)
    utils.map_qoi('P(dev) for nonvacant sites', results_both_df)
    utils.map_qoi('P(inventory) for homes built', results_both_df)
    utils.map_qoi('Mean underproduction', results_both_df)
    utils.map_qoi('RHNA Success', results_both_df)

    sea_plot = sea.histplot(results_both_df['P(dev) for nonvacant sites']).set_title(
        "Each city's P(dev) for nonvacant sites"
    )
    sea_plot.get_figure().savefig('./figures/Pdev_nonvacant.png')

    sea_plot = sea.histplot(results_both_df['P(dev) for vacant sites']).set_title("Each city's P(dev) for vacant sites")
    sea_plot.get_figure().savefig('./figures/Pdev_vacant.png')

    sea_plot = sea.histplot(results_both_df['P(dev) for vacant sites']).set_title("Each city's P(dev)")
    sea_plot.get_figure().savefig('./figures/Pdev.png')

    sea_plot = sea.histplot(results_both_df['P(inventory) for homes built']).set_title("P(inventory) for homes built")
    sea_plot.get_figure().savefig('./figures/pinventory.png')

    sea_plot = sea.histplot(results_both_df['Mean underproduction']).set_title("Each city's mean underproduction")
    sea_plot.get_figure().savefig('./figures/mean_underproduction.png')

    sea_plot = sea.histplot(results_both_df['RHNA Success']).set_title("Each city's RHNA success")
    sea_plot.get_figure().savefig('./figures/rhna_succes.png')

    # Did RHNA success in last cycle actually have anything to do with how good the site inventory was?
    rhna_success = results_both_df['P(inventory) for homes built']
    p_dev = results_both_df['RHNA Success']

    is_null = np.isnan(rhna_success) | np.isnan(p_dev)

    sea_plot = sea.scatterplot(x=rhna_success[~is_null], y=p_dev[~is_null])
    sea_plot.set_title("Does RHNA success have anything to do with the realistic capacity of the inventory sites?")
    sea_plot.get_figure().savefig('./figures/did_realistic_capacity_calcs_matter.png')

    pdevs = results_both_df['P(dev) for inventory']
    rhnas = [utils.get_rhna_target(city) for city in results_both_df['City']]

    plt.figure(figsize=(5, 5))
    plt.scatter(rhnas, pdevs, s=10, alpha=0.7)
    plt.xlabel("RHNA Allocation")
    plt.ylabel("Pdev")
    plt.title("Do smaller rhna allocations contribute to high P(dev)s?")
    plt.savefig('./figures/rhna_vs_pdev.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cache', action='store_true')
    args = parser.parse_args()

    if args.use_cache:
        with open('cities_with_sites_cache.pkl', 'rb') as f:
            cities_with_sites = pickle.load(f)

        with open('cities_with_permits_cache.pkl', 'rb') as f:
            cities_with_permits = pickle.load(f)
    else:
        cities_with_sites, cities_with_permits = load_sites_and_permits()

        with open('cities_with_sites_cache.pkl', 'wb') as f:
            pickle.dump(cities_with_sites, f)

        with open('cities_with_permits_cache.pkl', 'wb') as f:
            pickle.dump(cities_with_permits, f)

    # Dump match results to JSON, for use in website
    print("Creating JSON output for map...")
    map_utils.write_matches_to_files(cities_with_sites, cities_with_permits, Path('./map_results'))

    cities = sorted(set(cities_with_sites.keys()) & set(cities_with_permits.keys()))
    assert len(cities) == 97

    # Add an "overall" row so that we have the overall stats in the final table
    overall_sites = pd.concat([cities_with_sites[city] for city in cities])
    overall_permits = pd.concat([cities_with_permits[city] for city in cities])
    cities_with_sites['Overall'] = overall_sites
    cities_with_permits['Overall'] = overall_permits
    cities.append('Overall')

    print("Getting APN results...")
    apn_results_df = pd.DataFrame(
        parallel_process(
            get_results_for_city_kwargs,
            [
                dict(city=city, sites=cities_with_sites[city], permits=cities_with_permits[city], match_by='apn')
                for city in cities
            ],
        )
    )

    print("Getting geo results...")
    results_geo_df = pd.DataFrame(
        parallel_process(
            get_results_for_city_kwargs,
            [
                dict(city=city, sites=cities_with_sites[city], permits=cities_with_permits[city], match_by='geo')
                for city in cities
            ],
        )
    )

    print("Getting geo lax results...")
    results_geo_lax_df = pd.DataFrame(
        parallel_process(
            get_results_for_city_kwargs,
            [
                dict(city=city, sites=cities_with_sites[city], permits=cities_with_permits[city], match_by='geo', geo_matching_lax=True)
                for city in cities
            ],
        )
    )

    print("Getting apn or geo results...")
    results_both_df = pd.DataFrame(
        parallel_process(
            get_results_for_city_kwargs,
            [
                dict(city=city, sites=cities_with_sites[city], permits=cities_with_permits[city], match_by='both')
                for city in cities
            ],
        )
    )

    print("Getting apn or geo lax results...")
    results_both_lax_df = pd.DataFrame(
        parallel_process(
            get_results_for_city_kwargs,
            [
                dict(city=city, sites=cities_with_sites[city], permits=cities_with_permits[city], match_by='both', geo_matching_lax=True)
                for city in cities
            ],
        )
    )

    apn_results_df.to_csv('results/apn_matching_results.csv', index=False)
    results_geo_df.to_csv('results/geo_matching_results.csv', index=False)
    results_geo_lax_df.to_csv('results/geo_matching_lax_results.csv', index=False)
    results_both_df.to_csv('results/apn_or_geo_matching_results.csv', index=False)
    results_both_lax_df.to_csv('results/apn_or_geo_matching_lax_results.csv', index=False)

    combined_df = (
        apn_results_df
        .merge(results_geo_df, on='City', suffixes=[' (by APN)', ' (by conservative geomatching)'])
        .merge(results_geo_lax_df, on='City', suffixes=['', ' (by lenient geomatching)'])
        .merge(results_both_df, on='City', suffixes=['', ' (by APN or conservative geomatching)'])
        .merge(results_both_lax_df, on='City', suffixes=['', ' (by APN or lenient geomatching)'])
    )
    combined_df.to_csv('results/combined_df.csv', index=False)


    make_plots(results_both_df.query('City != "Overall"'))

    ground_truth_cities = ['Los Altos', 'San Francisco', 'San Jose']
    ground_truth_results_df = pd.DataFrame([get_ground_truth_results_for_city(city) for city in ground_truth_cities])
    ground_truth_results_df.to_csv('results/ground_truth_results.csv', index=False)

    # Additional summary stats for results section
    get_additional_stats(results_both_df.query('City != "Overall"')).to_csv('results/overall_summary_stats.csv', index=False)
    get_additional_stats(results_both_lax_df.query('City != "Overall"')).to_csv('results/overall_summary_stats_lax.csv', index=False)


def find_n_matches_raw_apn(cities):
    n_matches = 0
    for city in cities:
        site_df = utils.load_site_inventory(city, standardize_apn=False)
        permits_df = utils.load_all_new_building_permits(city, standardize_apn=False)
        city_matches, _, _ = utils.calculate_pdev_for_inventory(site_df, permits_df, 'apn')
        n_matches += city_matches
    return n_matches


def find_city_where_apn_formatting_mattered(cities):
    for city in cities:
        site_raw = utils.load_site_inventory(city, standardize_apn=False)
        permits_raw = utils.load_all_new_building_permits(city, standardize_apn=False)
        n_matches_raw, _, _ = utils.calculate_pdev_for_inventory(site_raw, permits_raw, 'apn')

        site_cln = utils.load_site_inventory(city, standardize_apn=True)
        permits_cln = utils.load_all_new_building_permits(city, standardize_apn=True)
        n_matches_cln, _, _ = utils.calculate_pdev_for_inventory(site_cln, permits_cln, 'apn')

        if n_matches_raw != n_matches_cln:
            matching_cln_site_indexes = site_cln[site_cln.apn.isin(permits_cln.apn)].index
            matching_raw_site_indexes = site_raw[site_raw.apn.isin(permits_raw.apn)].index
            new_matches_idx = list(set(matching_cln_site_indexes) - set(matching_raw_site_indexes))
            print(city)
            for idx in new_matches_idx:
                apn = site_cln.loc[idx].apn
                print('Clean sites apn', apn)
                print('Raw sites apn', site_raw.loc[idx].apn)
                permit_idx = permits_cln[permits_cln.apn == apn].index[0]
                print('Clean permits apn', permits_cln.loc[permit_idx].apn)
                print('Raw permits apn', permits_raw.loc[permit_idx].apn)
            break

def plot_pdev_vs_vacant_land(results_both_df):
    to_plot = pd.wide_to_long(results_both_df, stubnames='P(dev)', i=['City'], j='Vacant', suffix='.*')
    to_plot = to_plot.reset_index("Vacant")
    to_plot = to_plot[~(to_plot['Vacant'] == ' for inventory')]
    to_plot = to_plot.replace({' for nonvacant sites': 'nonvacant',
                               ' for vacant sites': 'vacant'})

    to_barplot = to_plot.copy()
    to_barplot['P(dev)'] = (to_plot['P(dev)'].values / .2).round(0) / 5
    to_barplot['P(dev)'] = to_barplot['P(dev)'].astype(str)
    to_barplot = to_barplot[to_barplot['P(dev)'] != 'nan']

    p_map = {'0.0' : '0 ≤ p < .2',
         '0.2': '.2 ≤ p < .4',
         '0.4': '.4 ≤ p < .6',
         '0.6': '.6 ≤ p < .8',
         '0.8': '.8 ≤ p ≤ 1',
         '1.0':  '.8 ≤ p ≤ 1'}
    to_barplot = to_barplot.replace(p_map)

    order_ps = ['0 ≤ p < .2',
                '.2 ≤ p < .4',
                '.4 ≤ p < .6',
                '.6 ≤ p < .8',
                '.8 ≤ p ≤ 1']
    sea.set(font_scale=1.1)
    plt.figure(figsize=(8, 6))
    ax = sea.countplot(x=to_barplot['P(dev)'], hue=to_barplot['Vacant'], data=to_barplot, order=order_ps)
    plt.legend(loc='upper right', title='Land')
    plt.title("Distribution of P(dev) Across Cities")
    plt.savefig('./figures/pdev_vs_vacancy.jpg')

def plot_pdev_vs_inventory_size(results_both_df, cities_with_sites, cities_with_permits):
    city_n_sites = {}

    for city, sites in cities_with_sites.items():
        if city in cities_with_permits:
            city_n_sites[city] = len(sites.index)

    # Include inventory size into results_df
    n_sites_df = pd.DataFrame.from_dict(city_n_sites, orient='index', columns=['n_sites'])
    n_sites_df = n_sites_df.reset_index()
    n_sites_df = n_sites_df.rename({'index': 'City'}, axis=1)
    combined_df = results_both_df.merge(n_sites_df, on='City')

    # Plot
    sea.set()
    plt.figure(figsize=(8, 6))
    ax = sea.regplot(x=combined_df['n_sites'], y=combined_df['P(dev) for inventory'], truncate=True, robust=True)
    ax.set(xscale="log")
    ax.set_ylim((-0.1, 1.1))
    plt.title('P(dev) as a function of site inventory size')
    plt.ylabel("P(dev) for City's Inventory")
    plt.xlabel("# of Sites in City's Inventory")
    plt.savefig('./figures/pdev_vs_inventory_size.jpg')

if __name__ == '__main__':
    main()
