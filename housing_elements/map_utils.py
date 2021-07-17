from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import shutil
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from tqdm import tqdm
from housing_elements.parallel_utils import parallel_process

from . import utils


# def permits_group_to_dicts(group_df: pd.DataFrame) -> list:


def add_matches_dict_to_sites(sites: gpd.GeoDataFrame, permits: gpd.GeoDataFrame, matches: Matches) -> gpd.GeoDataFrame:
    permits_renamed_df = (
        permits[['permyear', 'address', 'totalunit', 'hcategory']]
        .rename(columns={
            'permyear': 'permit_year',
            'address': 'permit_address',
            'totalunit': 'permit_units',
            'hcategory': 'permit_category',
        })
        # Make the order deterministic to avoid spurious diffs
        .sort_values(['permit_year', 'permit_address', 'permit_units', 'permit_category'])
        .replace({np.nan: None})  # for JSON reasons
        .rename_axis('permits_index')
        .reset_index()
    )

    output_df = sites[[
        'relcapcty', 'geometry', 'sitetype', 'is_vacant', 'is_nonvacant'
    ]].rename(columns={'relcapcty': 'site_capacity_units'})

    output_df['sites_index'] = output_df.index

    for buffer in [0, 5, 10, 25, 50, 75, 100]:
        matches_df = utils.get_matches_df(
            matches, matching_logic=utils.MatchingLogic(match_by='both', geo_matching_buffer=f'{buffer}ft'), add_match_type_labels=True
        )

        match_results_col = f'match_results_{buffer}ft'
        apn_match_col = f'apn_matched_{buffer}ft'
        geo_match_col = f'geo_matched_{buffer}ft'

        if len(matches_df):
            merged = matches_df.merge(
                permits_renamed_df,
                how='left',
                on='permits_index'
            )
            merged['match_type'] = np.select(
                [
                    merged['apn_matched'] & merged['geo_matched'],
                    merged['apn_matched'],
                    merged['geo_matched']
                ],
                [
                    'apn, geo',
                    'apn',
                    'geo',
                ],
                default=None
            )


            merged = merged.rename(columns={
                'apn_matched': apn_match_col,
                'geo_matched': geo_match_col,
            })

            match_results_by_site = (
                merged
                .groupby('sites_index')
                .apply(lambda group: group.drop(columns=['sites_index']).to_dict('records'))
                .reset_index(name=match_results_col)
            )

            apn_matched_by_site = merged.groupby('sites_index')[apn_match_col].any().reset_index()
            geo_matched_by_site = merged.groupby('sites_index')[geo_match_col].any().reset_index()

            output_df = output_df.merge(
                match_results_by_site,
                on='sites_index',
                how='left',
            ).merge(
                apn_matched_by_site,
                on='sites_index',
                how='left',
            ).merge(
                geo_matched_by_site,
                on='sites_index',
                how='left',
            )

            output_df[apn_match_col] = output_df[apn_match_col].fillna(False)
            output_df[geo_match_col] = output_df[geo_match_col].fillna(False)
        else:
            output_df[match_results_col] = None
            output_df[apn_match_col] = False
            output_df[geo_match_col] = False

    output_df = output_df.drop(columns=['sites_index'])

    for buffer in [5, 10, 25, 50, 75, 100]:
        assert (output_df['apn_matched_0ft'] == output_df[f'apn_matched_{buffer}ft']).all()

    output_df = output_df.drop(
        columns=[f'apn_matched_{buffer}ft' for buffer in [5, 10, 25, 50, 75, 100]]
    ).rename(
        columns={'apn_matched_0ft': 'apn_matched'}
    )

    return output_df


def _to_geojson_dict(row: pd.Series) -> dict:
    """
    Ugh, I have to write my own helper to turn a row into GeoJSON, since GeoPandas/fiona
    gets mad if your properties are not scalars.
    """
    return {
        'type': 'Feature',
        'geometry': shapely.geometry.mapping(row['geometry']),
        'properties': row.drop('geometry').to_dict()
    }

def write_geodataframe_to_geojson(df: gpd.GeoDataFrame, path: Path) -> None:
    # Browsers can't read JSON with NaN values.
    # I think None would be serialized as None, which is allowed.
    if len(df) == 0:
        raise ValueError("Cannot write empty GeoDataFrame")

    df = df.replace({np.nan: None})

    json_value = {
        'type': 'FeatureCollection',
        'features': df.apply(_to_geojson_dict, axis='columns').tolist()
    }

    with path.open('w') as f:
        json.dump(json_value, f)

def process_city(city: str, sites: gpd.GeoDataFrame, permits: gpd.GeoDataFrame, matches: utils.Matches, output_dir: str) -> dict:
    sites_with_matches_df = add_matches_dict_to_sites(sites, permits, matches)

    formatted_permits_df = permits[['permyear', 'address', 'totalunit', 'hcategory', 'geometry']].rename(
        columns={
            'permyear': 'permit_year',
            'address': 'permit_address',
            'totalunit': 'permit_units',
            'hcategory': 'permit_category',
        }
    )
    # We can't plot these permits with failed geomatching anyway, so might as well drop them
    # (we get an error when trying to call `shapely.geometry.mapping` on None).
    formatted_permits_df = formatted_permits_df.dropna(subset=['geometry'])

    city_path = Path(output_dir, city)
    city_path.mkdir(parents=True, exist_ok=True)
    write_geodataframe_to_geojson(sites_with_matches_df, Path(output_dir, city, 'sites_with_matches.geojson'))
    write_geodataframe_to_geojson(formatted_permits_df, Path(output_dir, city, 'permits.geojson'))

    # Save the map bounds of the city
    min_lng, min_lat, max_lng, max_lat = sites.geometry.total_bounds
    bounds = [
        (min_lng, min_lat), (max_lng, max_lat)
    ]

    # Kinda dumb that we're returning this from here... whatever, it's fine for now
    return {
        'city': city,
        'bounds': bounds,
    }

def process_city_kwargs(kwargs: dict) -> dict:
    return process_city(**kwargs)


def write_matches_to_files(
    cities_with_sites: Dict[str, gpd.GeoDataFrame],
    cities_with_permits: Dict[str, pd.DataFrame],
    output_dir: Path,
    all_matches: Dict[str, utils.Matches] = None,
) -> None:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    summary_info = []

    cities = set(cities_with_sites.keys()) & set(cities_with_permits.keys())

    summary_info = parallel_process(
        process_city_kwargs,
        [
            dict(
                city=city,
                sites=cities_with_sites[city],
                permits=cities_with_permits[city],
                matches=all_matches[city],
                output_dir=output_dir,
            )
            for city in cities
         ]
    )

    summary_df = pd.DataFrame(summary_info)

    apn_results_df = pd.read_csv(f'results/apn_matching_results.csv')
    apn_results_df_formatted = format_results_df(apn_results_df)
    summary_df[f'results_apn_only'] = summary_df['city'].map(apn_results_df_formatted)

    for buffer in [0, 5, 10, 25, 50, 75, 100]:
        results_df = pd.read_csv(f'results/apn_or_geo_matching_{buffer}ft_results.csv')
        results_df_formatted = format_results_df(results_df)
        summary_df[f'results_apn_and_geo_{buffer}ft'] = summary_df['city'].map(results_df_formatted)

    summary_df.to_json(Path(output_dir, 'summary.json'), orient='records')

    # Get rid of the numpy int/float types by round-tripping via JSON/dicts. Ugh this is so annoying
    non_numpy_summary_df = pd.DataFrame.from_records(json.loads(summary_df.to_json(orient='records')))

    city_shapes_df = gpd.read_file('data/raw_data/bay_area_map/bay.shp')[['city', 'geometry']]
    city_shapes_df['city'] = city_shapes_df['city'].str.title()
    city_shapes_df = city_shapes_df.merge(
        non_numpy_summary_df,
        on='city'
    )
    write_geodataframe_to_geojson(city_shapes_df, Path(output_dir, 'summary.geojson'))

    # Same as summary.geojson, except the geometry is the centroid of each polygon rather than
    # the whole shape. This is needed so that we can add labels to each city in Mapbox, and have the
    # label be in the middle of the polygon. For whatever reason, Mapbox can't calculate the centroid
    # itself, and it does this annoying thing where it labels each polygon in multiple places if you
    # give it a polygon to add a text label to.
    city_centroids_df = city_shapes_df.copy()
    city_centroids_df.geometry = city_centroids_df.geometry.centroid
    write_geodataframe_to_geojson(city_centroids_df, Path(output_dir, 'summary_centroids.geojson'))


def format_results_df(results_df: pd.DataFrame) -> Dict[str, Any]:
    results = {}
    for _, row in results_df.iterrows():
        city = row['City']
        results[city] = {
            'overall': {
                'P(dev)': row['P(dev) for inventory'],
                '# matches': row['# matches'],
            },
            'vacant': {
                'P(dev)': row['P(dev) for vacant sites'],
                '# matches': row['# vacant matches'],
            },
            'nonvacant': {
                'P(dev)': row['P(dev) for nonvacant sites'],
                '# matches': row['# nonvacant matches'],
            },
        }

    return results


def get_match_stats(matches_df: pd.DataFrame) -> Dict[str, Dict[str, Union[float, int]]]:
    return {
        'apn': _match_stats(matches_df['apn_matched']),
        'geo': _match_stats(matches_df['geo_matched']),
        'either': _match_stats(matches_df['apn_matched'] | matches_df['geo_matched']),
        'geo_lax': _match_stats(matches_df['geo_matched_lax']),
        'either_lax': _match_stats(matches_df['apn_matched'] | matches_df['geo_matched_lax']),
    }

def _match_stats(match_indicators: pd.Series) -> Dict[str, Union[float, int]]:
    return {
        'fraction': match_indicators.mean(),
        'sites': len(match_indicators),
        'matches': match_indicators.sum(),
    }
