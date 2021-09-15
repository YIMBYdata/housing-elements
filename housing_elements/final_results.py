from __future__ import annotations
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from collections import Counter
from housing_elements import analysis_utils, plot_utils, data_loading_utils, map_utils, los_altos_permits, san_francisco_permits, san_jose_permits
from pathlib import Path
import warnings
import os, sys
import pickle
import argparse
from housing_elements.parallel_utils import parallel_process

# Silence an annoying warning that I get when running pd.read_excel
warnings.filterwarnings("ignore", message="Data Validation extension is not supported and will be removed")

WEBSITE_DATA_PATH = Path('./website/public/data')


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def load_sites_and_permits():
    sites_df = data_loading_utils.load_all_sites()
    all_cities = sites_df.jurisdict.unique()

    assert len(all_cities) == 108

    cities_with_sites = {}
    for city in all_cities:
        with HiddenPrints():
            try:
                sites = data_loading_utils.load_site_inventory(city)
                assert sites.shape[0]
                cities_with_sites[city] = sites
            except Exception:
                print("Loading sites failed for " + city, file=sys.stderr)

    assert len(cities_with_sites) == 106

    cities_with_permits = {}
    for city in all_cities:
        with HiddenPrints():
            try:
                cities_with_permits[city] = data_loading_utils.load_all_new_building_permits(city)
            except Exception:
                print("Loading permits failed for " + city, file=sys.stderr)

    assert len(cities_with_permits) == 99
    assert len(set(cities_with_permits).intersection(set(cities_with_sites))) == 97

    cities = sorted(set(cities_with_sites.keys()) & set(cities_with_permits.keys()))

    overall_sites = pd.concat([cities_with_sites[city] for city in cities])
    overall_permits = pd.concat([cities_with_permits[city] for city in cities], ignore_index=True)
    cities_with_sites['Overall'] = overall_sites
    cities_with_permits['Overall'] = overall_permits

    return cities_with_sites, cities_with_permits

def cached_load_sites_and_permits(use_cache: bool) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if use_cache:
        with open('cities_with_sites_cache.pkl', 'rb') as f:
            cities_with_sites = pickle.load(f)

        with open('cities_with_permits_cache.pkl', 'rb') as f:
            cities_with_permits = pickle.load(f)

        with open('ground_truth_cities_with_permits_cache.pkl', 'rb') as f:
            ground_truth_cities_with_permits = pickle.load(f)
    else:
        cities_with_sites, cities_with_permits = load_sites_and_permits()

        ground_truth_cities_with_permits = {
            'San Jose': san_jose_permits.load_all_permits(),
            'San Francisco': san_francisco_permits.load_all_permits(),
            'Los Altos': los_altos_permits.load_all_permits(),
        }
        for city in ground_truth_cities_with_permits:
            ground_truth_cities_with_permits[city] = ground_truth_permits_post_processing(ground_truth_cities_with_permits[city], city)

        with open('cities_with_sites_cache.pkl', 'wb') as f:
            pickle.dump(cities_with_sites, f)

        with open('cities_with_permits_cache.pkl', 'wb') as f:
            pickle.dump(cities_with_permits, f)

        with open('ground_truth_cities_with_permits_cache.pkl', 'wb') as f:
            pickle.dump(ground_truth_cities_with_permits, f)

    return cities_with_sites, cities_with_permits, ground_truth_cities_with_permits

def get_results_for_city_kwargs(kwargs):
    with HiddenPrints():
        return get_results_for_city(**kwargs)

def get_results_for_city(
    city: str,
    sites: gpd.GeoDataFrame,
    permits: gpd.GeoDataFrame,
    matches: Matches,
    matching_logic: MatchingLogic,
) -> pd.DataFrame:
    """
    :param geo_matching_buffer: Options are '5ft', '10ft', '25ft', '50ft', '75ft', '100ft'.
    """
    nonvacant_matches, nonvacant_sites, nonvacant_ratio = analysis_utils.calculate_pdev_for_nonvacant_sites(
        sites, matches, matching_logic
    )
    vacant_matches, vacant_sites, vacant_ratio = analysis_utils.calculate_pdev_for_vacant_sites(
        sites, matches, matching_logic
    )
    all_matches, all_sites, all_ratio = analysis_utils.calculate_pdev_for_inventory(
        sites, matches, matching_logic
    )

    # Ground truth datasets won't have these income-related columns
    has_bmr_info = 'vlowndr' in permits.columns
    if has_bmr_info:
        bmr_matches, bmr_permits, p_inventory_bmr_units = analysis_utils.calculate_pinventory_for_dev_bmr_units(
            permits, matches, matching_logic
        )
        bmr_match_formatted = f'{bmr_matches} / {bmr_permits}'
    else:
        p_inventory_bmr_units = None
        bmr_match_formatted = None

    return {
        'City': city,
        'Units permitted (2015-2019)': permits.totalunit.sum(),
        'Mean underproduction': analysis_utils.calculate_underproduction_on_sites(
            sites, permits, matches, matching_logic
        ),
        'Units built to units claimed ratio on matched sites': analysis_utils.calculate_city_unit_ratio(
            sites, permits, matches, matching_logic
        ),
        'RHNA Success': analysis_utils.calculate_rhna_success(city, permits),
        'Units permitted / claimed capacity': analysis_utils.calculate_permits_to_capacity_ratio(sites, permits),
        'Units permitted via BPS / claimed capacity': analysis_utils.calculate_permits_to_capacity_ratio_via_bps(sites, city),
        'P(inventory) for homes built': analysis_utils.calculate_pinventory_for_dev(
            permits, matches, matching_logic
        ),
        'P(inventory) for projects built': analysis_utils.calculate_pinventory_for_dev_by_project(
            permits, matches, matching_logic
        ),
        'P(inventory) for BMR units': p_inventory_bmr_units,
        '# BMR matches / # BMR permits': bmr_match_formatted,
        'P(dev) for nonvacant sites': nonvacant_ratio,
        'P(dev) for vacant sites': vacant_ratio,
        'P(dev) for inventory': all_ratio,
        '# nonvacant matches': f'{nonvacant_matches} / {nonvacant_sites}',
        '# vacant matches': f'{vacant_matches} / {vacant_sites}',
        '# matches': f'{all_matches} / {all_sites}',
    }

def ground_truth_permits_post_processing(permits: gpd.GeoDataFrame, city: str) -> gpd.GeoDataFrame:
    if 'geometry' in permits.columns:
        if isinstance(permits.geometry, gpd.GeoSeries):
            geometry = permits.geometry
        else:
            geometry = gpd.GeoSeries(permits['geometry'], index=permits.index, crs='EPSG:4326')
    else:
        geometry = gpd.GeoSeries(None, index=permits.index, crs='EPSG:4326')

    permits = gpd.GeoDataFrame(permits, geometry=geometry, crs='EPSG:4326')

    # Need to add this so that the raw APN matching code doesn't fail... even though it actually doesn't matter,
    # we're not going to look at the raw APN matches for ground truth datasets.
    permits['apn_raw'] = permits['apn'].astype(str)

    if city == 'San Jose':
        # the San Jose data already has "San Jose, CA" at the end
        address_suffix = None
    else:
        address_suffix = ', ' + city + ', CA'
    permits = data_loading_utils.impute_missing_geometries(permits, address_suffix)

    return permits


def get_ground_truth_results_for_city(
    city: str,
    sites: gpd.GeoDataFrame,
    permits: gpd.GeoDataFrame,
) -> pd.DataFrame:
    matches = analysis_utils.get_all_matches(sites, permits)
    return get_results_for_city(
        city, sites, permits, matches, analysis_utils.MatchingLogic(match_by='both', geo_matching_buffer='25ft', use_raw_apns=False)
    )


def get_additional_stats(results_df: pd.DataFrame, overall_row: pd.Series) -> str:
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
        sites_matches_col = results_df[match_cols[site_type]]
        num_matches = sites_matches_col.str.split('/').apply(lambda x: int(x[0]))
        num_sites = sites_matches_col.str.split('/').apply(lambda x: int(x[1]))

        p_dev_col = results_df[p_dev_cols[site_type]]

        results.append(
            {
                'Site type': site_type,
                'Overall development rate': '{:.1%}'.format(analysis_utils.adj_pdev(num_matches.sum() / num_sites.sum())),
                'Num sites': num_sites.sum(),
                'Median P(dev)': '{:.3f}'.format(analysis_utils.adj_pdev(p_dev_col.median())),
                'Mean P(dev)': '{:.3f}'.format(analysis_utils.adj_pdev(p_dev_col.mean())),
            }
        )

    output = ''
    output += '## Pdevs table (Table 1):\n'
    output += pd.DataFrame(results).to_csv(index=False)
    output += '\n'

    def add_stats(title, series, overall, extra_info=None):
        nonlocal output
        stats = get_summary_stats_for_series(series)
        output += title + ':\n'
        output += print_dict(stats)
        output += 'Overall: ' + str(overall) + '\n'
        if extra_info:
            output += extra_info
        output += '\n'

    add_stats(
        'adj P(dev) stats',
        analysis_utils.adj_pdev(results_df['P(dev) for inventory']),
        analysis_utils.adj_pdev(overall_row['P(dev) for inventory'])
    )
    add_stats(
        'adj P(dev) for vacant sites stats',
        analysis_utils.adj_pdev(results_df['P(dev) for vacant sites']),
        analysis_utils.adj_pdev(overall_row['P(dev) for vacant sites'])
    )
    add_stats(
        'adj P(dev) for nonvacant sites stats',
        analysis_utils.adj_pdev(results_df['P(dev) for nonvacant sites']),
        analysis_utils.adj_pdev(overall_row['P(dev) for nonvacant sites'])
    )

    add_stats(
        'Underproduction stats',
        results_df['Mean underproduction'],
        overall_row['Mean underproduction']
    )

    add_stats(
        'P(inventory) for homes built',
        results_df['P(inventory) for homes built'],
        overall_row['P(inventory) for homes built'],
    )

    add_stats(
        'P(inventory) for projects built',
        results_df['P(inventory) for projects built'],
        overall_row['P(inventory) for projects built'],
    )

    add_stats(
        'P(inventory) for BMR units',
        results_df['P(inventory) for BMR units'],
        overall_row['P(inventory) for BMR units'],
        extra_info='Number of cities with BMR units: {:d}\n'.format(results_df['P(inventory) for BMR units'].notnull().sum())
    )

    add_stats(
        '8/5 * RHNA success',
        8/5 * results_df['RHNA Success'],
        8/5 * overall_row['RHNA Success'],
    )

    add_stats(
        '8/5 * Units permitted / claimed capacity',
        8/5 * results_df['Units permitted / claimed capacity'],
        8/5 * overall_row['Units permitted / claimed capacity'],
    )

    output += '## Comparing buffer sizes (Table B.1 and B.2):\n\n'
    dfs = {
        'raw_apn': pd.read_csv('results/raw_apn_matching_results.csv'),
        'apn': pd.read_csv('results/apn_matching_results.csv'),
        'apn_or_geo_0ft': pd.read_csv('results/apn_or_geo_matching_0ft_results.csv'),
        'apn_or_geo_5ft': pd.read_csv('results/apn_or_geo_matching_5ft_results.csv'),
        'apn_or_geo_10ft': pd.read_csv('results/apn_or_geo_matching_10ft_results.csv'),
        'apn_or_geo_25ft': pd.read_csv('results/apn_or_geo_matching_25ft_results.csv'),
        'apn_or_geo_50ft': pd.read_csv('results/apn_or_geo_matching_50ft_results.csv'),
        'apn_or_geo_75ft': pd.read_csv('results/apn_or_geo_matching_75ft_results.csv'),
        'apn_or_geo_100ft': pd.read_csv('results/apn_or_geo_matching_100ft_results.csv'),
    }
    for matching_logic, df in dfs.items():
        cities_df = df.query('City != "Overall"')
        matching_logic_overall_row = df.set_index('City').loc['Overall']
        extra_info = f'# matches: {matching_logic_overall_row["# matches"]} ({eval(matching_logic_overall_row["# matches"]):.1%})\n'
        add_stats(
            f'adj P(dev) for {matching_logic}',
            analysis_utils.adj_pdev(cities_df['P(dev) for inventory']),
            analysis_utils.adj_pdev(matching_logic_overall_row['P(dev) for inventory']),
            extra_info
        )

    def format_percent(num: float) -> str:
        return '{:.1%}'.format(num)

    def format_mean_and_std(series: pd.Series) -> str:
        mean = series.mean()
        std = series.std()
        return f'{mean:.1%} (sd. {std:.2f})'

    output += 'P(dev) for ABAG as a whole, under different matching assumptions\n'
    for matching_logic, df in dfs.items():
        result = analysis_utils.adj_pdev(df.query('City == "Overall"')['P(dev) for inventory'].squeeze())
        output += matching_logic + f' {result:.3f}\n'

    output += '\n'

    output += 'P(dev) for median city, under different matching assumptions\n'
    for matching_logic, df in dfs.items():
        cities_df = df.query('City != "Overall"')
        result = analysis_utils.adj_pdev(cities_df['P(dev) for inventory'].median())
        output += matching_logic + f' {result:.3f}\n'

    output += '\n'

    output += 'Median P(inventory) for homes built, under different matching assumptions\n'
    for matching_logic, df in dfs.items():
        cities_df = df.query('City != "Overall"')
        output += matching_logic + ' ' + format_percent(cities_df['P(inventory) for homes built'].median()) + '\n'

    output += '\n'

    output += 'Median P(inventory) for projects built, under different matching assumptions\n'
    for matching_logic, df in dfs.items():
        cities_df = df.query('City != "Overall"')
        output += matching_logic + ' ' + format_percent(cities_df['P(inventory) for projects built'].median()) + '\n'
    output += '\n'

    output += 'P(inventory) for homes built for ABAG overall, under different matching assumptions\n'
    for matching_logic, df in dfs.items():
        result = df.query('City == "Overall"')['P(inventory) for homes built'].squeeze()
        output += matching_logic + ' ' + format_percent(result) + '\n'

    output += '\n'

    output += 'P(inventory) for projects built for ABAG overall, under different matching assumptions\n'
    for matching_logic, df in dfs.items():
        result = df.query('City == "Overall"')['P(inventory) for projects built'].squeeze()
        output += matching_logic + ' ' + format_percent(result) + '\n'
    output += '\n'

    output += 'Mean P(inventory) for homes built, under different matching assumptions\n'
    for matching_logic, df in dfs.items():
        cities_df = df.query('City != "Overall"')
        output += matching_logic + ' ' + format_mean_and_std(cities_df['P(inventory) for homes built']) + '\n'

    output += '\n'

    output += 'Mean P(inventory) for projects built, under different matching assumptions\n'
    for matching_logic, df in dfs.items():
        cities_df = df.query('City != "Overall"')
        output += matching_logic + ' ' + format_mean_and_std(cities_df['P(inventory) for projects built']) + '\n'
    output += '\n'

    output += make_bps_comparison_results()

    return output

def make_bps_comparison_results() -> None:
    cities_with_sites, cities_with_permits, _ = cached_load_sites_and_permits(use_cache=True)

    cities = set(cities_with_sites.keys()) & set(cities_with_permits.keys())
    cities -= analysis_utils.BPS_KNOWN_MISSING_CITIES
    cities.remove('Overall')

    results = []
    for city in cities:
        sites = cities_with_sites[city]
        permits = cities_with_permits[city]

        results.append({
            'city': city,
            'permits to capacity ratio (using ABAG dataset)': analysis_utils.calculate_permits_to_capacity_ratio(sites, permits),
            'permits to capacity ratio (using BPS dataset)': analysis_utils.calculate_permits_to_capacity_ratio_via_bps(sites, city),
        })

    results_df = pd.DataFrame(results)

    overall_sites = pd.concat([cities_with_sites[city] for city in cities])
    overall_permits = pd.concat([cities_with_permits[city] for city in cities])
    overall_row = pd.Series({
        'permits to capacity ratio (using ABAG dataset)': analysis_utils.calculate_permits_to_capacity_ratio(overall_sites, overall_permits),
        'permits to capacity ratio (using BPS dataset)': analysis_utils.calculate_permits_to_capacity_ratio_via_bps(overall_sites, list(cities)),
    })

    output = 'BPS comparison table results:\n'
    def add_stats(title, series, overall, extra_info=None):
        nonlocal output
        stats = get_summary_stats_for_series(series)
        output += title + ':\n'
        output += print_dict(stats)
        output += 'Overall: ' + str(overall) + '\n'
        if extra_info:
            output += extra_info
        output += '\n'

    add_stats(
        '8/5 * permits to capacity ratio (using ABAG dataset)',
        8/5 * results_df['permits to capacity ratio (using ABAG dataset)'],
        8/5 * overall_row['permits to capacity ratio (using ABAG dataset)'],
    )

    add_stats(
        '8/5 * permits to capacity ratio (using BPS dataset)',
        8/5 * results_df['permits to capacity ratio (using BPS dataset)'],
        8/5 * overall_row['permits to capacity ratio (using BPS dataset)'],
    )

    return output


def get_summary_stats_for_series(series: pd.Series) -> Dict[str, float]:
    return {
        'Mean': series.mean(),
        'Stddev': series.std(),
        'Median': series.median(),
        'Min': series.min(),
        '25th percentile': series.quantile(0.25),
        '75th percentile': series.quantile(0.75),
        'Max': series.max(),
    }

def print_dict(results_dict: Dict[str, Any]) -> str:
    return '\n'.join([f'{k}: {v}' for k, v in results_dict.items()]) + '\n'


def make_plots(results_both_df: pd.DataFrame) -> None:
    plot_utils.map_qoi('P(dev) for inventory', results_both_df)
    plot_utils.map_qoi('P(dev) for vacant sites', results_both_df)
    plot_utils.map_qoi('P(dev) for nonvacant sites', results_both_df)
    plot_utils.map_qoi('P(inventory) for homes built', results_both_df)
    plot_utils.map_qoi('Mean underproduction', results_both_df)
    plot_utils.map_qoi('RHNA Success', results_both_df)

    plt.figure()
    plot_inventory_permits_by_year()

    plt.figure()
    plot_pdev_vs_vacant_land(results_both_df)

    plt.figure()
    sea_plot = sea.histplot(results_both_df['P(dev) for nonvacant sites']).set_title(
        "Each city's P(dev) for nonvacant sites"
    )
    sea_plot.get_figure().savefig('./figures/Pdev_nonvacant_histogram.png', dpi=500)

    plt.figure()
    sea_plot = sea.histplot(results_both_df['P(dev) for vacant sites']).set_title("Each city's P(dev) for vacant sites")
    sea_plot.get_figure().savefig('./figures/Pdev_vacant_histogram.png', dpi=500)

    plt.figure()
    sea_plot = sea.histplot(results_both_df['P(dev) for vacant sites']).set_title("Each city's P(dev)")
    sea_plot.get_figure().savefig('./figures/Pdev_histogram.png', dpi=500)

    plt.figure()
    sea_plot = sea.histplot(results_both_df['P(inventory) for homes built'])
    sea_plot.set(xlabel='Share of homes built on inventory sites', ylabel='Number of Cities')
    fname = './results/csvs_for_plots/pinventory_histogram.csv'
    results_both_df[['City', 'P(inventory) for homes built']].to_csv(fname)
    sea_plot.get_figure().savefig('./figures/pinventory_histogram.png', dpi=500)

    plt.figure()
    sea_plot = sea.histplot(results_both_df['Mean underproduction']).set_title("Each city's mean underproduction")
    sea_plot.get_figure().savefig('./figures/mean_underproduction_histogram.png', dpi=500)

    plt.figure()
    sea_plot = sea.histplot(results_both_df['RHNA Success']).set_title("Each city's RHNA success")
    sea_plot.get_figure().savefig('./figures/rhna_success_histogram.png', dpi=500)

    # Did RHNA success in last cycle actually have anything to do with how good the site inventory was?
    rhna_success = results_both_df['P(inventory) for homes built']
    p_dev = results_both_df['RHNA Success']

    is_null = np.isnan(rhna_success) | np.isnan(p_dev)

    plt.figure()
    sea_plot = sea.scatterplot(x=rhna_success[~is_null], y=p_dev[~is_null])
    sea_plot.set_title("Does RHNA success have anything to do with the P(dev) of the inventory sites?")
    sea_plot.get_figure().savefig('./figures/did_realistic_capacity_calcs_matter_scatterplot.png', dpi=500)

    pdevs = results_both_df['P(dev) for inventory']
    rhnas = [data_loading_utils.get_rhna_target(city) for city in results_both_df['City']]

    plt.figure(figsize=(5, 5))
    plt.scatter(rhnas, pdevs, s=10, alpha=0.7)
    plt.xlabel("RHNA Allocation")
    plt.ylabel("Pdev")
    plt.title("Do smaller rhna allocations contribute to high P(dev)s?")
    plt.savefig('./figures/rhna_vs_pdev_scatterplot.png', dpi=500)
    
    plot_utils.make_cover()


def get_all_matches_kwargs(kwargs):
    return analysis_utils.get_all_matches(**kwargs)

def create_results_csv_files(
    cities: List[str],
    cities_with_sites: Dict[str, gpd.GeoDataFrame],
    cities_with_permits: Dict[str, gpd.GeoDataFrame],
    all_matches: Dict[str, pd.DataFrame],
):
    print("Getting APN results...")
    apn_results_df = pd.DataFrame(
        parallel_process(
            get_results_for_city_kwargs,
            [
                dict(
                    city=city,
                    sites=cities_with_sites[city],
                    permits=cities_with_permits[city],
                    matches=all_matches[city],
                    matching_logic=analysis_utils.MatchingLogic(match_by='apn'),
                )
                for city in cities
            ],
        )
    )

    print("Getting APN raw results...")
    raw_apn_results_df = pd.DataFrame(
        parallel_process(
            get_results_for_city_kwargs,
            [
                dict(
                    city=city,
                    sites=cities_with_sites[city],
                    permits=cities_with_permits[city],
                    matches=all_matches[city],
                    matching_logic=analysis_utils.MatchingLogic(
                        match_by='apn',
                        use_raw_apns=True
                    ),
                )
                for city in cities
            ],
        )
    )

    apn_results_df.to_csv('results/apn_matching_results.csv', index=False)
    raw_apn_results_df.to_csv('results/raw_apn_matching_results.csv', index=False)

    for buffer in ['0ft', '5ft', '10ft', '25ft', '50ft', '75ft', '100ft']:
        print(f"Getting geo {buffer} results...")
        geo_results_df = pd.DataFrame(
            parallel_process(
                get_results_for_city_kwargs,
                [
                    dict(
                        city=city,
                        sites=cities_with_sites[city],
                        permits=cities_with_permits[city],
                        matches=all_matches[city],
                        matching_logic=analysis_utils.MatchingLogic(
                            match_by='geo',
                            geo_matching_buffer=buffer
                        )
                    )
                    for city in cities
                ],
            )
        )
        geo_results_df.to_csv(f'results/geo_matching_{buffer}_results.csv', index=False)

        print(f"Getting apn or geo {buffer} results...")
        both_results_df = pd.DataFrame(
            parallel_process(
                get_results_for_city_kwargs,
                [
                    dict(
                        city=city,
                        sites=cities_with_sites[city],
                        permits=cities_with_permits[city],
                        matches=all_matches[city],
                        matching_logic=analysis_utils.MatchingLogic(
                            match_by='both',
                            geo_matching_buffer=buffer
                        )
                    )
                    for city in cities
                ],
            )
        )
        both_results_df.to_csv(f'results/apn_or_geo_matching_{buffer}_results.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-sites-permits-cache', action='store_true')
    parser.add_argument('--use-matches-cache', action='store_true')
    parser.add_argument('--additional-results-only', action='store_true')
    parser.add_argument('--plots-only', action='store_true')
    parser.add_argument('--ground-truth-results-only', action='store_true')
    parser.add_argument('--map-results-only', action='store_true')
    args = parser.parse_args()

    if args.additional_results_only:
        print_summary_stats()
        return

    if args.plots_only:
        make_plots(pd.read_csv('results/apn_or_geo_matching_25ft_results.csv').query('City != "Overall"'))
        return

    cities_with_sites, cities_with_permits, ground_truth_cities_with_permits = cached_load_sites_and_permits(args.use_sites_permits_cache)

    cities = sorted(set(cities_with_sites.keys()) & set(cities_with_permits.keys()))
    assert len(cities) == 98

    # load_sites_and_permits should have added an 'Overall' DataFrame to both, which includes sites and permits across all cities that have both present.
    assert 'Overall' in cities

    # Move 'Overall' to the end of the list
    cities.remove('Overall')
    cities.append('Overall')

    if args.ground_truth_results_only:
        get_ground_truth_results(cities_with_sites, ground_truth_cities_with_permits)
        return

    if args.use_matches_cache:
        print("Loading all matches from cache...")
        with open('all_matches_cache.pkl', 'rb') as f:
            all_matches = pickle.load(f)
    else:
        print("Computing all matches...")
        all_matches = parallel_process(
            get_all_matches_kwargs,
            [{'sites': cities_with_sites[city], 'permits': cities_with_permits[city]} for city in cities]
        )
        all_matches = dict(zip(cities, all_matches))

        with open('all_matches_cache.pkl', 'wb') as f:
            pickle.dump(all_matches, f)

    if args.map_results_only:
        map_utils.write_matches_to_files(cities_with_sites, cities_with_permits, WEBSITE_DATA_PATH, all_matches=all_matches)
        return

    create_results_csv_files(cities, cities_with_sites, cities_with_permits, all_matches)

    # Dump match results to JSON, for use in website
    print("Creating JSON output for map...")
    map_utils.write_matches_to_files(cities_with_sites, cities_with_permits, WEBSITE_DATA_PATH, all_matches=all_matches)

    # 25ft is the chosen buffer size
    results_df = pd.read_csv('results/apn_or_geo_matching_25ft_results.csv')

    make_plots(results_df.query('City != "Overall"'))

    get_ground_truth_results(cities_with_sites, ground_truth_cities_with_permits)

    print_summary_stats()

def get_ground_truth_results(cities_with_sites: Dict[str, gpd.GeoDataFrame], ground_truth_cities_with_permits: gpd.GeoDataFrame) -> None:
    ground_truth_cities = ['Los Altos', 'San Francisco', 'San Jose']

    ground_truth_results_df = pd.DataFrame([
        get_ground_truth_results_for_city(city, cities_with_sites[city], ground_truth_cities_with_permits[city]) for city in ground_truth_cities
    ])
    ground_truth_results_df.to_csv('results/ground_truth_results.csv', index=False)

def print_summary_stats():
    # 25ft is the chosen buffer
    results_df = pd.read_csv('results/apn_or_geo_matching_25ft_results.csv')
    results_upper_bound_df = pd.read_csv('results/apn_or_geo_matching_100ft_results.csv')

    # Additional summary stats for results section
    Path('results/overall_summary_stats.csv').write_text(
        get_additional_stats(
            results_df.query('City != "Overall"'),
            results_df.set_index('City').loc['Overall'],
        )
    )

    get_final_appendix_table(results_df, results_upper_bound_df).to_csv('results/final_appendix_table.csv', index=False)

    make_ground_truth_summary_table()


def make_ground_truth_summary_table():
    # Make comparison table for ground truth
    results_df = pd.read_csv('results/apn_or_geo_matching_25ft_results.csv')
    ground_truth_results_df = pd.read_csv('results/ground_truth_results.csv')

    ground_truth_summary_df = pd.DataFrame({
        'City': ground_truth_results_df['City'],
        'P(dev) for inventory, 8 years, ground truth': analysis_utils.adj_pdev(ground_truth_results_df['P(dev) for inventory']),
    })
    abag_pdevs = results_df.set_index('City')['P(dev) for inventory'].apply(analysis_utils.adj_pdev).to_dict()
    ground_truth_summary_df['P(dev) for inventory, 8 years, ABAG'] = ground_truth_summary_df['City'].map(abag_pdevs)

    ground_truth_summary_df.to_csv('results/ground_truth_summary.csv', index=False)



def get_final_appendix_table(results_df, results_upper_bound_df):
    assert len(results_df) == len(results_upper_bound_df)
    assert (results_df['City'] == results_upper_bound_df['City']).all()

    df = pd.DataFrame({
        'City': results_df['City']
    })

    df['P(dev) for inventory sites'] = results_df['P(dev) for inventory'].apply(analysis_utils.adj_pdev).apply('{:.1%}'.format)
    df['P(dev) for inventory sites (upper bound estimate)'] = results_upper_bound_df['P(dev) for inventory'].apply(analysis_utils.adj_pdev).apply('{:.1%}'.format)
    df['Citywide production relative to claimed capacity'] = (8/5 * results_df['Units permitted / claimed capacity']).apply('{:.1%}'.format)
    df['Realized vs. anticipated density on inventory sites'] = results_df['Mean underproduction'].dropna().apply('{:.2f}'.format).reindex(df.index, fill_value='N/A')
    df['Permitted units on inventory sites, as fraction of all permitted units'] = (
        results_df['P(inventory) for homes built']
    ).apply('{:.0%}'.format)

    df = df[
        df['City'] != 'Overall'
    ]

    return df



def find_n_matches_raw_apn(cities):
    n_matches = 0
    for city in cities:
        site_df = data_loading_utils.load_site_inventory(city, standardize_apn=False)
        permits_df = data_loading_utils.load_all_new_building_permits(city, standardize_apn=False)
        city_matches, _, _ = data_loading_utils.calculate_pdev_for_inventory(site_df, permits_df, 'apn')
        n_matches += city_matches
    return n_matches


def find_city_where_apn_formatting_mattered(cities):
    for city in cities:
        site_raw = data_loading_utils.load_site_inventory(city, standardize_apn=False)
        permits_raw = data_loading_utils.load_all_new_building_permits(city, standardize_apn=False)
        n_matches_raw, _, _ = data_loading_utils.calculate_pdev_for_inventory(site_raw, permits_raw, 'apn')

        site_cln = data_loading_utils.load_site_inventory(city, standardize_apn=True)
        permits_cln = data_loading_utils.load_all_new_building_permits(city, standardize_apn=True)
        n_matches_cln, _, _ = data_loading_utils.calculate_pdev_for_inventory(site_cln, permits_cln, 'apn')

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
    to_save = to_barplot[['P(dev)', 'Vacant']]
    to_save.to_csv('./results/csvs_for_plots/histogram_pdev_disaggregated_by_sitetype.csv')
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
    plt.legend(loc='upper right', title='Parcels')
    plt.ylabel("Number of Cities")
    plt.savefig('./figures/histogram_pdev_disaggregated_by_sitetype.jpg', dpi=500)


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
    plt.savefig('./figures/pdev_vs_inventory_size.jpg', dpi=500)


def analyze_realcap_input(cities):
    """
    :param: cities: list of strings of city names

    Return tuple of number of inventory sites, number of sites with realistic capacity
    that can't be converted to a number without special handling, and return the number of
    sites that have no realistic capacity listed whatsoever.
    """
    n_sites, n_missing, n_parse_fail, n_unlisted = 0, 0, 0, 0

    for city in cities:
        df = data_loading_utils.load_site_inventory(city, fix_realcap=False)
        n_missing += df.relcapcty.isna().sum()
        n_unlisted += df.realcap_not_listed.sum()
        n_parse_fail += df.realcap_parse_fail.sum()
        n_sites += len(df.index)

    assert n_missing == (n_unlisted + n_parse_fail)
    return n_sites, n_parse_fail, n_unlisted


def plot_inventory_permits_by_year():
    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    with open(f'{path}/cities_with_sites_cache.pkl', 'rb') as f:
        cities_with_sites = pickle.load(f)

    with open(f'{path}/cities_with_permits_cache.pkl', 'rb') as f:
        cities_with_permits = pickle.load(f)

    with open(f'{path}/all_matches_cache.pkl', 'rb') as f:
        matches = pickle.load(f)

    cities = [c for c in cities_with_permits if c in cities_with_sites]

    match_df = None
    for city in cities:
        permits = cities_with_permits[city]
        match_city = analysis_utils.get_matches_df(matches[city], analysis_utils.MatchingLogic("both", "25ft"))
        match_city['permyear'] = permits.loc[match_city.permits_index].permyear.values
        match_city.sort_values('permyear', inplace=True)
        match_city.drop_duplicates('sites_index', inplace=True)
        if match_df is None:
            match_df = match_city
        else:
            match_df = pd.concat((match_df, match_city))

    match_df.drop_duplicates('sites_index', inplace=True)
    permyears = match_df.permyear.values
    py = [p for p in permyears if p > 2014]
    ordered_py = [(years, counts) for years, counts in Counter(py).items()]
    years, counts = [c[0] for c in ordered_py], [c[1] for c in ordered_py]
    pd.DataFrame({"years":years, "counts": counts}).to_csv(path + "/results/csvs_for_plots/permits_by_year.csv")
    sea.set()
    plt.bar(years, counts)
    plt.ylabel("Number of Permits")
    plt.xlabel("Year")
    plt.title("When do inventory sites get permitted?")
    path = path + "/figures/permits_by_year.jpg"
    plt.savefig(path, dpi=500)

if __name__ == '__main__':
    main()
