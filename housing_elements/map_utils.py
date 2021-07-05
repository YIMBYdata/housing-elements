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
import folium
from tqdm import tqdm

from . import utils

def shapely_polygon_to_coords(shape: shapely.geometry.Polygon) -> List[Tuple[float, float]]:
    return [(lat, lng) for lng, lat in shape.exterior.coords]

def _dedupe_matches(apn_matches: Optional[List[dict]], geo_matches: Optional[List[dict]]) -> List[dict]:
    apn_matches = apn_matches or []
    geo_matches = geo_matches or []

    # (Possibly unnecessary) optimization
    if not apn_matches and not geo_matches:
        return []

    # Super inefficient algorithm, but each of these lists should be length <5 so it should be fine.
    apn_tuples = {(row['permyear'], row['address'], row['totalunit'], row['hcategory']) for row in apn_matches}
    geo_tuples = {(row['permyear'], row['address'], row['totalunit'], row['hcategory']) for row in geo_matches}

    def make_match_dict(permit_tuple: Tuple[int, str, float], match_types: List[str]) -> dict:
        # Probably would be neater to do this column renaming in `combine_match_dfs`, but whatever,
        # it's easier to do it here.
        return {
            'permit_year': permit_tuple[0],
            'permit_address': permit_tuple[1],
            'permit_units': permit_tuple[2],
            'permit_category': permit_tuple[3],
            'match_type': match_types,
        }

    deduped_matches = []
    for permit_tuple in apn_tuples:
        if permit_tuple in geo_tuples:
            deduped_matches.append(make_match_dict(permit_tuple, ['apn', 'geo']))
            geo_tuples.remove(permit_tuple)
        else:
            deduped_matches.append(make_match_dict(permit_tuple, ['apn']))

    for permit_tuple in geo_tuples:
        deduped_matches.append(make_match_dict(permit_tuple, ['geo']))

    # Make the order deterministic, to avoid spurious diffs
    deduped_matches = sorted(deduped_matches, key=sort_key)

    return deduped_matches

def sort_key(permit_dict):
    keys = ['permit_address', 'permit_year', 'permit_units', 'permit_category', 'match_type']
    return tuple([(permit_dict[key] is not None, permit_dict[key]) for key in keys])

def combine_match_dfs(sites: gpd.GeoDataFrame, permits: gpd.GeoDataFrame, matches: Matches) -> gpd.GeoDataFrame:

    flattened_permits = (
        permits[['permyear', 'address', 'totalunit', 'hcategory']]
        .replace({np.nan: None})  # for JSON reasons
        .to_dict(orient='records')
    )
    flattened_permits = dict(zip(permits.index, flattened_permits))

    output_df = sites[['relcapcty', 'geometry', 'sitetype']]
    output_df = output_df.rename(columns={'relcapcty': 'site_capacity_units'})

    def make_matches_series(matches_df, index):
        if len(matches_df) == 0:
            return pd.Series(None, index)
        return (
            matches_df
            .groupby('sites_index')
            .apply(lambda group: [flattened_permits[i] for i in group['permits_index']])
        )

    # print(matches)

    output_df['apn_match_results'] = make_matches_series(matches.apn_matches, output_df.index)
    output_df['geo_match_results'] = make_matches_series(matches.geo_matches, output_df.index)
    output_df['geo_match_results_lax'] = make_matches_series(matches.geo_matches_lax, output_df.index)
    for col in ['apn_match_results', 'geo_match_results', 'geo_match_results_lax']:
        output_df[col] = output_df[col].replace({np.nan: None})

    output_df['apn_matched'] = output_df['apn_match_results'].notnull()
    output_df['geo_matched'] = output_df['geo_match_results'].notnull()
    output_df['geo_matched_lax'] = output_df['geo_match_results_lax'].notnull()

    output_df['match_results'] = output_df.apply(
        lambda row: _dedupe_matches(row['apn_match_results'], row['geo_match_results']),
        axis='columns'
    )

    output_df['match_results_lax'] = output_df.apply(
        lambda row: _dedupe_matches(row['apn_match_results'], row['geo_match_results_lax']),
        axis='columns'
    )

    output_df = (
        output_df.drop(columns=['apn_match_results', 'geo_match_results', 'geo_match_results_lax'])
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

def write_matches_to_files(
    cities_with_sites: Dict[str, gpd.GeoDataFrame],
    cities_with_permits: Dict[str, pd.DataFrame],
    output_dir: Path
) -> None:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    summary_info = []

    import itertools
    for city, sites in tqdm(cities_with_sites.items()):
        if len(sites) == 0:
            # This is the case for Orinda (not sure how this happened); not sure if there are other cities
            continue

        permits = cities_with_permits.get(city)
        if permits is not None:
            matches = utils.get_all_matches(sites, permits)
            matches_df = combine_match_dfs(sites, permits, matches)

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
            write_geodataframe_to_geojson(matches_df, Path(output_dir, city, 'sites_with_matches.geojson'))
            write_geodataframe_to_geojson(formatted_permits_df, Path(output_dir, city, 'permits.geojson'))

            # Save the map bounds of the city
            min_lng, min_lat, max_lng, max_lat = sites.geometry.total_bounds
            bounds = [
                (min_lng, min_lat), (max_lng, max_lat)
            ]

            summary_info.append({
                'city': city,
                'bounds': bounds,
                'overall_match_stats': get_match_stats(matches_df),
                'vacant_match_stats': get_match_stats(matches_df[matches_df['sitetype'] == 'Vacant']),
                'nonvacant_match_stats': get_match_stats(matches_df[matches_df['sitetype'] != 'Vacant']),
            })

    summary_df = pd.DataFrame(summary_info)
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
