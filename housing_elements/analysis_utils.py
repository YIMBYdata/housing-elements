from __future__ import annotations
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, NamedTuple
from housing_elements import data_loading_utils


def fraction_apns_nan(permits: pd.DataFrame) -> float:
    return permits.apn.isna().mean()


def has_more_than_q_real_apns(permits: pd.DataFrame, q: float) -> bool:
    """ Return list of cities with more than Q% real values.
    """
    assert 0 <= q <= 1, "q must be a fraction in [0, 1]"
    cutoff = 1 - q
    return fraction_apns_nan(permits) > cutoff


def calculate_pinventory_for_dev(
    permits: pd.DataFrame, matches: Matches, matching_logic: MatchingLogic
) -> float:
    """P(inventory|developed), over permitted units"""
    assert permits.index.nunique() == len(permits.index)

    matches_df = get_matches_df(matches, matching_logic)

    housing_on_sites = permits[permits.index.isin(matches_df['permits_index'])].totalunit.sum()
    total_units = permits.totalunit.sum()

    print("Units permitted on inventory sites:", housing_on_sites)
    print("Total units permitted:", total_units)

    if total_units:
        return housing_on_sites / total_units
    return np.nan


def filter_for_bmr_permits(permits: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if 'vlowdr' not in permits.columns:
        print(permits)
    return permits[
        # Must have low-income units
        (
            (permits['vlowdr'] > 0)
            | (permits['lowdr'] > 0)
            | (permits['vlowdr'] > 0)
            | (permits['lowdr'] > 0)
        )
        # Not a mixed-income/IZ or density bonus project
        & (permits['amodtot'] == 0)
        # exclude ADUs
        & (~permits['hcategory'].isin(['SU', 'ADU']))
        & (permits['totalunit'] > 1)
    ]


def calculate_pinventory_for_dev_bmr_units(
    permits: pd.DataFrame, matches: Matches, matching_logic: MatchingLogic
) -> float:
    """P(inventory|developed), over permitted units"""
    assert permits.index.nunique() == len(permits.index)

    matches_df = get_matches_df(matches, matching_logic)

    bmr_permits = filter_for_bmr_permits(permits)

    housing_on_sites = bmr_permits[bmr_permits.index.isin(matches_df['permits_index'])].totalunit.sum()
    total_units = bmr_permits.totalunit.sum()

    print("BMR units permitted on inventory sites:", housing_on_sites)
    print("Total BMR units permitted:", total_units)

    if total_units:
        return housing_on_sites / total_units
    return None


def calculate_pinventory_for_dev_by_project(
    permits: pd.DataFrame, matches: Matches, matching_logic: MatchingLogic
) -> float:
    """P(inventory|developed), over permitted projects"""
    assert permits.index.nunique() == len(permits.index)

    matches_df = get_matches_df(matches, matching_logic)

    return permits.index.isin(matches_df['permits_index']).mean()


def calculate_underproduction_on_sites(
    sites: pd.DataFrame, permits: pd.DataFrame, matches: Matches, matching_logic: MatchingLogic
) -> float:
    """
    Report the average ratio of (units built / units claimed) for all matched sites in the city.
    """
    assert sites.index.nunique() == len(sites.index)
    assert permits.index.nunique() == len(permits.index)

    matches_df = get_matches_df(matches, matching_logic).drop_duplicates()

    merged_df = sites[['relcapcty']].rename_axis('sites_index').reset_index().merge(
        matches_df,
        on='sites_index'
    ).merge(
        permits[['totalunit']].rename_axis('permits_index').reset_index(),
        on='permits_index'
    )

    # Get the match for each permit with the maximum claimed capacity
    deduped_df = merged_df.groupby(['sites_index', 'relcapcty'])['totalunit'].sum().reset_index()

    return deduped_df.query('relcapcty != 0').eval('totalunit / relcapcty').dropna().mean()


def calculate_city_unit_ratio(
    sites: pd.DataFrame, permits: pd.DataFrame, matches: Matches, matching_logic: MatchingLogic
) -> float:
    """
    Report the average ratio of (units built / units claimed) for all matched sites in the city.
    """
    assert sites.index.nunique() == len(sites.index)
    assert permits.index.nunique() == len(permits.index)

    matches_df = get_matches_df(matches, matching_logic).drop_duplicates()

    merged_df = sites[['relcapcty']].rename_axis('sites_index').reset_index().merge(
        matches_df,
        on='sites_index'
    ).merge(
        permits[['totalunit']].rename_axis('permits_index').reset_index(),
        on='permits_index'
    )

    merged_df = merged_df.dropna(subset=['totalunit', 'relcapcty']).query('relcapcty != 0')

    # Handle cases where one site matches multiple permits, or vice versa. Just make sure we're counting each once.
    total_units = merged_df.drop_duplicates('permits_index')['totalunit'].sum()
    total_capacity = merged_df.drop_duplicates('sites_index')['relcapcty'].sum()
    if total_capacity == 0 or pd.isnull(total_capacity):
        return None
    return total_units / total_capacity


def calculate_rhna_success(city: str, permits: pd.DataFrame) -> float:
    """Percentage of RHNA total built. Can exceed 100%.

    This is a crude proxy for RHNA success because it's insensitive to affordability levels.
    """
    total_units = permits.totalunit.sum()
    rhna_target = data_loading_utils.get_rhna_target(city)
    print("Total units permitted:", total_units)
    print("Total rhna target:", rhna_target)
    if rhna_target:
        return total_units / rhna_target
    return np.nan

def calculate_permits_to_capacity_ratio(sites: gpd.GeoDataFrame, permits: gpd.GeoDataFrame) -> float:
    return permits.totalunit.sum() / sites.relcapcty.sum()

BPS_KNOWN_MISSING_CITIES = {
    'Clayton',
    'Lafayette',
    'Moraga',
    'Saint Helena',
}

def calculate_permits_to_capacity_ratio_via_bps(sites: gpd.GeoDataFrame, city: Union[str, List[str]]) -> Optional[float]:
    if city == 'Overall':
        # I don't think we need the overall Bay Area numbers, and anyways how to do this is unclear:
        # do we include all cities in ABAG? All cities in our analysis? How do we deal with the 4 cities that
        # don't report data to BPS?
        return None

    bps_df = data_loading_utils.get_census_bps_dataset()
    if isinstance(city, list):
        rows_for_city = bps_df[
            bps_df['place_name'].isin(city)
        ]
        assert len(rows_for_city) == 5 * len(city)
    elif isinstance(city, str):
        rows_for_city = bps_df[
            bps_df['place_name'] == city
        ]
        if len(rows_for_city) == 0:
            if city in BPS_KNOWN_MISSING_CITIES:
                return None
            raise ValueError(f"City {city} not available in BPS dataset")

        if len(rows_for_city) != 5:
            raise ValueError(f"Not all years present for {city}: found {len(rows_for_city)} rows")

        assert set(rows_for_city['year'].unique()) == {'2015', '2016', '2017', '2018', '2019'}
    else:
        raise ValueError('city must be a str or List[str]')

    return rows_for_city['total_units'].sum() / sites.relcapcty.sum()


def merge_on_apn(sites: gpd.GeoDataFrame, permits: gpd.GeoDataFrame, use_raw_apns: bool = False) -> pd.DataFrame:
    """
    :return: a DataFrame with two columns, 'sites_index' and 'permits_index', indicating which rows
    in `sites` and `permits` were matched.
    """
    assert sites.index.nunique() == len(sites.index)
    assert permits.index.nunique() == len(permits.index)

    if use_raw_apns:
        apn_col = 'apn_raw'
        locapn_col = 'locapn_raw'
    else:
        apn_col = 'apn'
        locapn_col = 'locapn'

    sites = sites.rename_axis('sites_index').reset_index()[['sites_index', apn_col, locapn_col]]
    permits = permits.rename_axis('permits_index').reset_index()[['permits_index', apn_col]]

    merged_df_1 = sites.dropna(subset=[apn_col]).merge(
        permits,
        left_on=apn_col,
        right_on=apn_col,
    )

    merged_df_2 = sites.dropna(subset=[locapn_col]).merge(
        permits,
        left_on=locapn_col,
        right_on=apn_col,
    )

    return pd.concat([merged_df_1, merged_df_2])[['sites_index', 'permits_index']].drop_duplicates()


class Matches(NamedTuple):
    apn_matches: pd.DataFrame
    apn_matches_raw: pd.DataFrame
    geo_matches_0ft: pd.DataFrame
    geo_matches_5ft: pd.DataFrame
    geo_matches_10ft: pd.DataFrame
    geo_matches_25ft: pd.DataFrame
    geo_matches_50ft: pd.DataFrame
    geo_matches_75ft: pd.DataFrame
    geo_matches_100ft: pd.DataFrame


class MatchingLogic(NamedTuple):
    match_by: str
    geo_matching_buffer: str = '25ft'
    use_raw_apns: bool = False

def get_all_matches(sites: gpd.GeoDataFrame, permits: gpd.GeoDataFrame) -> Matches:
    """
    Helper function to precompute all matches. This should speed up all the downstream functions that depend on matching logic,
    so that we only compute them once.
    """
    apn_matches = merge_on_apn(sites, permits)
    apn_matches_raw = merge_on_apn(sites, permits, use_raw_apns=True)
    geo_matches_0ft = merge_on_address(sites, permits, buffer='0ft')
    geo_matches_5ft = merge_on_address(sites, permits, buffer='5ft')
    geo_matches_10ft = merge_on_address(sites, permits, buffer='10ft')
    geo_matches_25ft = merge_on_address(sites, permits, buffer='25ft')
    geo_matches_50ft = merge_on_address(sites, permits, buffer='50ft')
    geo_matches_75ft = merge_on_address(sites, permits, buffer='75ft')
    geo_matches_100ft = merge_on_address(sites, permits, buffer='100ft')

    return Matches(
        apn_matches,
        apn_matches_raw,
        geo_matches_0ft,
        geo_matches_5ft,
        geo_matches_10ft,
        geo_matches_25ft,
        geo_matches_50ft,
        geo_matches_75ft,
        geo_matches_100ft
    )

def get_matches_df(
    matches: Matches, matching_logic: MatchingLogic, add_match_type_labels: bool = False
) -> pd.DataFrame:
    """
    If match_by == 'both' and add_match_type_labels = True, it will add columns `apn_matched` and `geo_matched` to indicate
    for each row, which method was used to get the match.
    """
    if matching_logic.use_raw_apns:
        apn_df = matches.apn_matches_raw
    else:
        apn_df = matches.apn_matches

    if matching_logic.geo_matching_buffer == '0ft':
        geo_df = matches.geo_matches_0ft
    elif matching_logic.geo_matching_buffer == '5ft':
        geo_df = matches.geo_matches_5ft
    elif matching_logic.geo_matching_buffer == '10ft':
        geo_df = matches.geo_matches_10ft
    elif matching_logic.geo_matching_buffer == '25ft':
        geo_df = matches.geo_matches_25ft
    elif matching_logic.geo_matching_buffer == '50ft':
        geo_df = matches.geo_matches_50ft
    elif matching_logic.geo_matching_buffer == '75ft':
        geo_df = matches.geo_matches_75ft
    elif matching_logic.geo_matching_buffer == '100ft':
        geo_df = matches.geo_matches_100ft
    else:
        raise ValueError(f"Unknown geo_matching_buffer option: {matching_logic.geo_matching_buffer}")

    if matching_logic.match_by == 'apn':
        return apn_df
    elif matching_logic.match_by == 'geo':
        return geo_df
    elif matching_logic.match_by == 'both':
        # We need to make sure a permit isn't matched to multiple sites, i.e. one site via APN matching and one
        # via geo matching. So let's only consider a permit for geo-matching if it was not already matched by APN.
        #
        # This should not affect the results of P(dev), but it might affect the results of "achieved density vs.
        # claimed capacity". Not doing this deduping might bias that measure upward, because a permit might contribute
        # to "units built" on multiple sites.
        merged_apn_df = apn_df.assign(apn_matched=True).merge(
            geo_df.assign(geo_matched=True),
            on=['sites_index', 'permits_index'],
            how='left'
        )
        merged_apn_df['geo_matched'] = merged_apn_df['geo_matched'].fillna(False)

        geo_new_df = geo_df[
            ~geo_df['permits_index'].isin(apn_df['permits_index'])
        ].assign(apn_matched=False, geo_matched=True)

        result_df = pd.concat([merged_apn_df, geo_new_df])

        if not add_match_type_labels:
            result_df = result_df.drop(columns=['apn_matched', 'geo_matched'])

        return result_df
    else:
        raise ValueError(f"Unknown matching type: {matching_logic.match_by}")

def calculate_pdev_for_inventory(
    sites: pd.DataFrame, matches: Matches, matching_logic: MatchingLogic
) -> Tuple[int, int, float]:
    """
    Return tuple of (# matched permits, # total sites, P(permit | inventory_site))
    :param match_by: Can be 'apn', 'geo', or 'both'.
    """
    num_sites = len(sites)
    if num_sites == 0:
        return 0, 0, np.nan

    match_df = get_matches_df(matches, matching_logic)
    matched_site_indices = match_df['sites_index']

    is_match = sites.index.isin(matched_site_indices)

    return is_match.sum(), len(sites), is_match.mean()


def calculate_pdev_for_vacant_sites(
    sites: pd.DataFrame, matches: Matches, matching_logic: MatchingLogic
) -> Tuple[int, int, float]:
    """Return P(permit | inventory_site, vacant)"""
    vacant_rows = sites[sites.is_vacant].copy()
    return calculate_pdev_for_inventory(vacant_rows, matches, matching_logic)


def calculate_pdev_for_nonvacant_sites(
    sites: pd.DataFrame, matches: Matches, matching_logic: MatchingLogic
) -> Tuple[int, int, float]:
    """Return P(permit | inventory_site, non-vacant)"""
    nonvacant_rows = sites[sites.is_nonvacant].copy()
    return calculate_pdev_for_inventory(nonvacant_rows, matches, matching_logic)


def merge_on_address(sites: gpd.GeoDataFrame, permits: gpd.GeoDataFrame, buffer: str = '5ft') -> gpd.GeoDataFrame:
    """
    Returns all matches. Length of output is between 0 and len(permits). A site could be repeated, if it was matched with
    multiple permits.
    """
    assert sites.index.nunique() == len(sites.index)
    assert permits.index.nunique() == len(permits.index)

    sites = sites[['geometry']][sites['geometry'].notnull()].rename_axis('sites_index').reset_index()
    permits = permits[['geometry']][permits['geometry'].notnull()].rename_axis('permits_index').reset_index()

    # Switch to the most common projection for California. (It's in meters.)
    sites = sites.to_crs('EPSG:3310')
    permits = permits.to_crs('EPSG:3310')

    supported_buffers = [0, 5, 10, 25, 50, 75, 100]  # feet
    buffer_map = {f'{buffer}ft': buffer for buffer in supported_buffers}
    if buffer in buffer_map:
        buffer_feet = buffer_map[buffer]
    else:
        raise ValueError(f"Buffer option not recognized: {buffer}")

    # Buffer by N feet, and take the closest match for each permit, to limit false positives
    buffer_meters = buffer_feet * 0.3048
    permits_buffered = permits.copy()
    if buffer_meters != 0:
        permits_buffered.geometry = permits_buffered.geometry.buffer(buffer_meters)

    # Get all pairs of sites, permits within N meters of each other.
    # The geometry column will be taken the left dataframe (i.e. the site geometry).
    merged = gpd.sjoin(sites, permits_buffered, how='inner')

    # Add permit location as a column
    merged = merged.merge(
        permits.rename(columns={'geometry': 'permit_geometry'}),
        on='permits_index',
    )
    merged['distance'] = gpd.GeoSeries(merged['geometry']).distance(gpd.GeoSeries(merged['permit_geometry']))

    # Now, for each permit, only keep the closest site it matched to.
    merged = merged.sort_values(['permits_index', 'distance']).drop_duplicates(['permits_index'], keep='first').reset_index(drop=True)

    return merged[['sites_index', 'permits_index']].copy()


def adj_pdev(raw_pdev):
    if isinstance(raw_pdev, pd.Series):
        return raw_pdev.apply(adj_pdev)
    if np.isnan(raw_pdev):
        return np.nan
    assert 0 <= raw_pdev <= 1
    return min( 8 / 5 * raw_pdev, 1)

