import re
import os
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import seaborn as sea
import logging
from shapely.geometry import Point
from typing import List, Optional, Tuple
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import contextily as ctx
from housing_elements import geocode_cache
from matplotlib.colors import LinearSegmentedColormap

_logger = logging.getLogger(__name__)
ABAG = None
INVENTORY = None
TARGETS = None
XLSX_FILES = [
    ('Richmond', '2018'),
    ('PleasantHill', '2018'),
    ('Oakland', '2019'),
    ('Livermore', '2019'),
]
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

def load_apr_permits(
    city: str, year: str, filter_for_permits: bool = True
) -> pd.DataFrame:
    """
    :param year: must be either '2018' or '2019'
    """
    city = city.replace(" ", "")

    if (city, year) in XLSX_FILES:
        path = f"/data/raw_data/APRs/{city}{year}.xlsx"
    else:
        path = f"/data/raw_data/APRs/{city}{year}.xlsm"

    df = pd.read_excel(
        PARENT_DIR + path,
        sheet_name="Table A2",
        skiprows=10,
        usecols="A:AS",
        dtype={"Current APN": str},
    )
    df.columns = df.columns.str.replace("\s+", " ", regex=True).str.strip()

    # Delete these columns which are usually all null.
    often_null_columns = [
        "Prior APN+",
        "Local Jurisdiction Tracking ID+",
        "How many of the units were Extremely Low Income?+",
        "Infill Units? Y/N+",
        "Assistance Programs for Each Development (see instructions)",
        "Number of Demolished/Destroyed Units+",
        "Demolished or Destroyed Units+",
        "Demolished/Destroyed Units Owner or Renter+",
    ]
    for col in often_null_columns:
        if df[col].isnull().all():
            df = df.drop(columns=[col])
        else:
            num_not_null_rows = df[col].notnull().sum()
            _logger.info(
                f"Column {col} is not all null ({num_not_null_rows} rows not null), will not drop it."
            )

    # Sorry this is messy. Some say "enter 1", some say "enter 1000", but in either case we want to drop the column
    term_length_enter_1 = "Term of Affordability or Deed Restriction (years) (if affordable in perpetuity enter 1)+"
    term_length_enter_1000 = "Term of Affordability or Deed Restriction (years) (if affordable in perpetuity enter 1000)+"
    assert term_length_enter_1 in df.columns or term_length_enter_1000 in df.columns
    df.drop(
        columns=[
            term_length_enter_1,
            term_length_enter_1000,
        ],
        errors="ignore",
    )

    if filter_for_permits:
        df = df[df["Building Permits Date Issued"].notnull()].copy()

        # The excel file has unit numbers by affordability level for three categories:
        # planning entitlements, building permits, and certificates of occupancy.
        #
        # For affordability level X, that column will be present three times: "X", "X.1", "X.2"
        # for entitlements, permits, and COOs respectively. We only want the one for permits.
        income_categories = [
            "Very Low- Income Deed Restricted",
            "Very Low- Income Non Deed Restricted",
            "Low- Income Deed Restricted",
            "Low- Income Non Deed Restricted",
            "Moderate- Income Deed Restricted",
            "Moderate- Income Non Deed Restricted",
            "Above Moderate- Income",
        ]
        for category in income_categories:
            df = df.drop(columns=[category, category + ".2"])

        df = df.rename(
            columns={category + ".1": category for category in income_categories},
            errors="raise",
        )

        # Drop other entitlements/certificates of occupancy-related columns
        unrelated_columns = [
            "# of Units issued Entitlements",
            "Entitlement Date Approved",
            "Certificates of Occupancy or other forms of readiness (see instructions) Date Issued",
            "# of Units issued Certificates of Occupancy or other forms of readiness",
        ]
        for col in unrelated_columns:
            if not ((df[col] == 0) | (df[col].isnull())).all():
                num_rows = ((df[col] != 0) | df[col].notnull()).sum()
                _logger.info(
                    f"{num_rows} rows of {col} are not null or zero. Dropping the column anyway"
                )
            df = df.drop(columns=[col])

        return df
    else:
        return df

def fraction_apns_nan(permits: pd.DataFrame) -> float:
    return permits.apn.isna().mean()

def has_more_than_q_real_apns(permits: pd.DataFrame, q: float) -> bool:
    """ Return list of cities with more than Q% real values.
    """
    assert 0 <= q <= 1, "q must be a fraction in [0, 1]"
    cutoff = 1 - q
    return fraction_apns_nan(permits) > cutoff

def map_apr_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the column names in the spreadsheet to the one in the ABAG dataset, so they can be concatenated.
    """
    df = df.copy()
    mapping = {
        "Very Low- Income Deed Restricted": "vlowdr",
        "Very Low- Income Non Deed Restricted": "vlowndr",
        "Low- Income Deed Restricted": "lowdr",
        "Low- Income Non Deed Restricted": "lowndr",
        "Moderate- Income Deed Restricted": "moddr",
        "Moderate- Income Non Deed Restricted": "modndr",
        "Above Moderate- Income": "amodtot",
        "Project Name+": "projname",
        "Unit Category (SFA,SFD,2 to 4,5+,ADU,MH)": "hcategory",
        "Tenure R=Renter O=Owner": "tenure",
        "Street Address": "address",
        "Notes+": "notes",
        "Current APN": "apn",
        "# of Units Issued Building Permits": "totalunit",
    }
    df = df.rename(columns=mapping, errors="raise")

    # Deed-restricted X + Non-deed-restricted X = Total X (where X = very low, low, or moderate)
    df["vlowtot"] = df["vlowdr"] + df["vlowndr"]
    df["lowtot"] = df["lowdr"] + df["lowndr"]
    df["modtot"] = df["moddr"] + df["modndr"]

    df["permyear"] = pd.to_datetime(df["Building Permits Date Issued"]).dt.year

    return df


def load_abag_permits() -> gpd.GeoDataFrame:
    """
    Loads all 2013-2017 building permits from ABAG as a GeoDataFrame.
    """
    global ABAG
    if ABAG is None:
        geometry_df = gpd.read_file(PARENT_DIR + "/data/raw_data/abag_building_permits/permits.shp")
        data_df = pd.read_csv(PARENT_DIR + "/data/raw_data/abag_building_permits/permits.csv")

        # There shouldn't be any rows with geometry data that don't have label data
        assert geometry_df["joinid"].isin(data_df["joinid"]).all()

        ABAG = gpd.GeoDataFrame(data_df.merge(geometry_df, how="left", on="joinid"))
    return ABAG

def impute_missing_geometries(df: gpd.GeoDataFrame, address_suffix: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Fills in the missing entries in the input GeoDataFrame's 'geometry' field,
    using the 'address' column.

    The input GeoDataFrame's projection must be the standard (long, lat)
    projection (ESPG:4326).

    :param address_suffix: (Optional.) A string to add to the end of the address (e.g. the city name)
        to help the geocoder find the address, in case the city/state name isn't already part of the
        address field.
    """
    assert df.crs.to_epsg() == 4326

    # Some rows with 'address' being null might also be missing, but we don't have an address to
    # geocode, so too bad.
    missing_indices = df[
        df.geometry.isnull()
        & df['address'].notnull()
        & (df['address'].apply(type) == str)
    ].index

    addresses = df.loc[missing_indices]['address']

    if address_suffix:
        addresses = addresses + address_suffix

    missing_points_geoseries = geocode_results_to_geoseries(
        geocode_cache.lookup(addresses),
        index=missing_indices
    )

    fixed_df = df.copy()
    fixed_df.geometry = df.geometry.combine_first(missing_points_geoseries)

    return fixed_df


def impute_missing_geometries_from_file(df: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Use a county dataset to impute APNs with missing geometries.
    """
    missing_rows = df[
        df.geometry.isnull()
    ][['apn']].drop_duplicates()

    merged = missing_rows.merge(
        parcels[['apn', 'geometry']],
        how='left',
        on='apn',
    )

    df_copy = df.copy()

    # merged should have the same indices as the original DataFrame
    df_copy.geometry = df_copy.geometry.combine_first(merged.geometry)

    return df_copy


def load_all_new_building_permits(city: str) -> pd.DataFrame:
    """
    Returns the combined dataset of 2013-2019 permits, combining the 2013-2017 dataset from ABAG with the 2018-19 dataset from the APRs.
    """

    abag_permits_df = load_abag_permits()

    assert city in set(abag_permits_df["jurisdictn"])
    abag_permits_df = abag_permits_df[abag_permits_df["jurisdictn"] == city].copy()

    apr_permits_2018_df = load_apr_permits(city, "2018")
    apr_permits_2019_df = load_apr_permits(city, "2019")

    apr_permits_df = map_apr_column_names(
        pd.concat([apr_permits_2018_df, apr_permits_2019_df])
    )

    permits_df = pd.concat(
        [
            abag_permits_df,
            apr_permits_df,
        ]
    ).reset_index(drop=True)

    permits_df = standardize_apn_format(permits_df, 'apn')

    # We need to add "<city name>, CA" to the addresses when we're geocoding them because the ABAG dataset (as far as I've seen)
    # doesn't have the city name or zip code. Otherwise, we get a bunch of results of that address from all over the US.
    return impute_missing_geometries(permits_df, address_suffix=f', {city}, CA')


def load_all_sites() -> gpd.GeoDataFrame:
    global INVENTORY
    if INVENTORY is None:
        INVENTORY = gpd.read_file(
            PARENT_DIR + "/data/raw_data/housing_sites/xn--Bay_Area_Housing_Opportunity_Sites_Inventory__20072023_-it38a.shp"
        )
    return INVENTORY

def load_rhna_targets() -> str:
    global TARGETS
    if TARGETS is None:
        TARGETS = pd.read_csv(PARENT_DIR + '/data/raw_data/rhna_targets.txt', sep=', ', engine='python')
    return TARGETS

def load_site_inventory(city: str, exclude_approved_sites: bool = True) -> pd.DataFrame:
    """
    Return the 5th RHNA cycle site inventory for CITY.

    :param exclude_approved_sites:
        Whether to exclude sites with sitetype = 'Approved' (i.e. sites that already had
        planning entitlements before the start of the 5th RHNA cycle).
        These sites have a higher probability of development (i.e. something very close to 1) than a typical site,
        and therefore including these would bias the estimates upward.
    """
    sites_df = load_all_sites()

    assert (
        city in sites_df.jurisdict.values
    ), "city must be a jurisdiction in the inventory. Be sure to capitalize."

    rows_to_keep = sites_df.eval(f'jurisdict == "{city}" and rhnacyc == "RHNA5"').fillna(False)
    #print(rows_to_keep)
    if exclude_approved_sites:
        # Keep sites where sitetype is null or sitetype != Approved.
        # I guess it's possible that some null rows are also pre-approved, but whatever. We can
        # document that as a potential data issue.
        rows_to_keep &= (sites_df['sitetype'] != 'Approved').fillna(True)
        #print(rows_to_keep)

    sites = sites_df[rows_to_keep].copy()
    sites.fillna(value=np.nan, inplace=True)

    if city in ('Oakland', 'Los Altos Hills', 'Napa County', 'Newark'):
        sites = remove_range_in_realcap(sites)
    if city in ('Danville', 'San Ramon', 'Corte Madera', 'Portola Valley'):
        sites = remove_units_in_realcap(sites)
    if city == 'El Cerrito':
        sites = fix_el_cerrito_realcap(sites)

    is_null_realcap = sites.relcapcty.isna()
    sites['realcap_not_listed'] = is_null_realcap
    sites['relcapcty'] = pd.to_numeric(sites['relcapcty'], errors='coerce')
    sites['realcap_parse_fail'] = sites.relcapcty.isna() & ~is_null_realcap


    sites = drop_constant_cols(sites)
    sites = add_cols_for_sitetype(sites)
    sites = standardize_apn_format(sites, 'apn')
    sites = standardize_apn_format(sites, 'locapn')
    print("DF shape", sites.shape)

    if city == 'San Francisco':
        # Exclude PDR sites that have a stated capacity of zero.
        # According to SF's website, "In order to protect PDR, residential development would be prohibited,
        # while office, retail, and institutional uses (schools, hospitals, etc.) would be limited.
        # HOWEVER, residences, offices and retail which currently exist legally in these areas may stay indefinitely."
        sites = sites[
            sites['relcapcty'] != 0
        ]
    return sites

def add_cols_for_sitetype(sites):
    vacant = ['Underutilized and Va', 'Vacant', 'Undeveloped', 'Open Space', 'Underutilized & Vaca', 'Vacant and Underutil']
    nonvacant = ['Opportunity', 'Underused site', 'Underutilized, margi', 'Non-Vacant', "Infill", 'underutilize', 'Underutilized']
    sites['is_vacant'] = sites.sitetype.isin(vacant)
    sites['is_nonvacant'] = sites.sitetype.isin(nonvacant)
    sites['na_vacant'] = ~sites.sitetype.isin(vacant + nonvacant)
    return sites

def standardize_apn_format(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if not is_numeric_dtype(df[column].dtype):
        df[column] = df[column].str.replace("-", "", regex=False)
        df[column] = df[column].str.replace(" ", "", regex=False)
        df[column] = df[column].str.replace(r"[a-zA-Z|.+,;:/]", '', regex=True)

        # Some extra logic to handle Oakland, Hayward, Pittsburg, San Bruno, South SF, Windsor
        df[column] = df[column].str.replace("\n", "", regex=False)
        df[column] = df[column].where(df[column].str.isdigit())
        df[column] = df[column].where(df[column].str.len() > 0)
        df[column] = df[column].where(df[column].str.len() <= 23)

    df[column] = df[column].dropna().astype('int64').astype('Int64').reindex(df.index, fill_value=pd.NA)
    return df

def drop_constant_cols(sites: pd.DataFrame) -> pd.DataFrame:
    """Return df with constant columns dropped unless theyre necessary for QOI calculations."""
    if len(sites.index) > 1:
        dont_drop = ['existuse', 'totalunit', 'permyear', 'relcapcty', 'apn', 'sitetype',
                     'realcap_parse_fail', 'realcap_not_listed']
        is_constant = ((sites == sites.iloc[0]).all())
        constant_cols = is_constant[is_constant].index.values
        constant_cols = list(set(constant_cols) - set(dont_drop))
        print("Dropping constant columns:", constant_cols)
        return sites.drop(constant_cols, axis=1)
    return sites

def remove_range_in_realcap(sites: pd.DataFrame) -> pd.DataFrame:
    # E.g. Oakland, Newark
    sites.relcapcty = sites.relcapcty.str.split('-').str[-1]
    # Los Altos Hills
    sites.relcapcty = sites.relcapcty.str.split(' to ').str[-1]
    return sites

def remove_units_in_realcap(sites: pd.DataFrame) -> pd.DataFrame:
    # San Ramon
    sites.relcapcty = sites.relcapcty.str.replace('á', '', regex=False)
    # Danville
    sites.relcapcty = sites.relcapcty.str.replace('sfr', '', regex=False)
    sites.relcapcty = sites.relcapcty.str.replace('SFR', '', regex=False)
    sites.relcapcty = sites.relcapcty.str.replace('mfr', '', regex=False)
    # Danville, Corte Madera, Portola Valley
    sites.relcapcty = sites.relcapcty.str.split(' ').str[0]
    return sites

def fix_el_cerrito_realcap(sites: pd.DataFrame) -> pd.DataFrame:
    """El Cerrito's realcap is in plain english, listing primary units and accessory units."""
    el_cerrito_rc = []
    for v in sites.relcapcty.values:
        # If realcap includes primary and accessory units
        if isinstance(v, str) and 'primary and' in v:
            # Then let realcap equal double the # of primary units (which is always true)
            v = int(v.split(' ')[0]) * 2
        el_cerrito_rc.append(v)
    sites.relcapcty = el_cerrito_rc
    sites.relcapcty = sites.relcapcty.str.split(' ').str[0]
    return sites

def calculate_pinventory_for_dev(
    sites: pd.DataFrame, permits: pd.DataFrame
) -> float:
    """P(inventory|developed)"""
    housing_on_sites = permits[permits.apn.isin(sites.apn)].totalunit.sum()
    total_units = permits.totalunit.sum()

    print("Units permitted on inventory sites:", housing_on_sites)
    print("Total units permitted:", total_units)

    if total_units:
        return housing_on_sites / total_units
    return np.nan


def calculate_underproduction_on_sites(
    sites: pd.DataFrame, permits: pd.DataFrame
) -> float:
    """For each inventory site that was built, report underproduction relative to units promised."""
    inventory_sites_permitted = permits[permits.apn.isin(sites.apn) | permits.apn.isin(sites.locapn)]
    inventory_sites_permitted = inventory_sites_permitted.apn.unique()
    if not len(inventory_sites_permitted):
        return np.nan
    units_built_over_claimed = []
    for site_apn in inventory_sites_permitted:
        n_claimed_by_apn = sites[sites.apn == site_apn].relcapcty.sum()
        n_claimed_by_locapn = sites[sites.locapn == site_apn].relcapcty.sum()
        units_claimed = np.nanmax(np.array((n_claimed_by_apn), n_claimed_by_locapn))
        units_built = permits[permits.apn == site_apn].totalunit.sum()
        if units_claimed:
            units_built_over_claimed.append(units_built / units_claimed)
    if units_built_over_claimed:
        return sum(units_built_over_claimed) / len(units_built_over_claimed)
    return np.nan

def calculate_rhna_success(city: str, permits: pd.DataFrame) -> float:
    """Percentage of RHNA total built. Can exceed 100%.

    This is a crude proxy for RHNA success because it's insensitive to affordability levels.
    """
    total_units = permits.totalunit.sum()
    rhna_target = get_rhna_target(city)
    print("Total units permitted:", total_units)
    print("Total rhna target:", rhna_target)
    if rhna_target:
        return total_units / rhna_target
    return np.nan

def get_rhna_target(city: str) -> float:
    rhna_targets = load_rhna_targets()
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        rhna_target = rhna_targets.query('City == @city')['Total'].values[0]
    return rhna_target


def merge_on_apn(sites: gpd.GeoDataFrame, permits: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    merged_df_1 = sites.merge(
        permits,
        left_on='apn',
        right_on='apn',
        how='left',
    )

    merged_df_2 = sites.merge(
        permits,
        left_on='locapn',
        right_on='apn',
        how='left',
    )

    return [merged_df_1, merged_df_2]


def calculate_pdev_for_inventory(
    sites: pd.DataFrame, permits: pd.DataFrame, match_by: str = 'apn', geo_matching_lax: bool = False
) -> Tuple[int, int, float]:
    """
    Return tuple of (# matched permits, # total sites, P(permit | inventory_site))
    :param match_by: Can be 'apn', 'geo', or 'both'.
    """
    num_sites = len(sites)
    if num_sites == 0:
        return 0, 0, np.nan

    if match_by not in ['apn', 'geo', 'both']:
        raise ValueError(f"Parameter match_by={match_by} not recognized. must equal 'apn', 'geo', or 'both'.")

    sites = sites.copy()
    sites['index'] = pd.RangeIndex(len(sites))

    match_dfs = []
    if match_by in ['apn', 'both']:
        # Select just a few columns before merging, because it makes it wayyy faster
        match_dfs.extend(merge_on_apn(sites[['index', 'apn', 'locapn']], permits[['apn', 'permyear']]))

    if match_by in ['geo', 'both']:
        merged_df = merge_on_address(
            sites[['index', 'apn', 'geometry']],
            permits[['apn', 'permyear', 'geometry']],
            lax=geo_matching_lax
        )
        match_dfs.append(merged_df)

    # Add a copy of each site, with empty matches, as our "default" output row for each site if it was not matched.
    null_rows = sites[['index', 'apn', 'locapn']]

    match_df = pd.concat(match_dfs + [null_rows])

    # dedupe, keeping the one that is merged
    match_df = match_df.sort_values('permyear', na_position='last').drop_duplicates(['index'], keep='first')

    assert len(match_df) == len(sites)

    is_match = match_df['permyear'].notnull()

    return is_match.sum(), len(sites), is_match.mean()


def calculate_pdev_for_vacant_sites(sites: pd.DataFrame, permits: pd.DataFrame, match_by: str = 'apn') -> Tuple[int, int, float]:
    """Return P(permit | inventory_site, vacant)"""
    vacant_rows = sites[sites.is_vacant].copy()
    return calculate_pdev_for_inventory(vacant_rows, permits, match_by)


def calculate_pdev_for_nonvacant_sites(sites: pd.DataFrame, permits: pd.DataFrame, match_by: str = 'apn') -> Tuple[int, int, float]:
    """Return P(permit | inventory_site, non-vacant)"""
    nonvacant_rows = sites[sites.is_nonvacant].copy()
    return calculate_pdev_for_inventory(nonvacant_rows, permits, match_by)


def merge_on_address(sites: gpd.GeoDataFrame, permits: gpd.GeoDataFrame, lax: bool = False) -> gpd.GeoDataFrame:
    """
    Returns all matches. Length of output is between 0 and len(permits). A site could be repeated, if it was matched with
    multiple permits.
    """
    sites = sites[sites['geometry'].notnull()].copy().reset_index(drop=True)
    permits = permits[permits['geometry'].notnull()].copy().reset_index(drop=True)

    sites['index_sites'] = pd.RangeIndex(len(sites))
    permits['index_permits'] = pd.RangeIndex(len(permits))

    # Switch to the most common projection for California. (It's in meters.)
    sites = sites.to_crs('EPSG:3310')
    permits = permits.to_crs('EPSG:3310')

    if lax:
        # Buffer by 15 meters, which is about 50 feet, and take the closest match for each permit, to limit false positives
        permits_buffered = permits.copy()
        permits_buffered.geometry = permits_buffered.geometry.buffer(15)

        # Get all pairs of sites, permits within 15 meters of each other. The geometry column will be the left dataframe by default
        # (so the site geometry).
        merged = gpd.sjoin(sites, permits_buffered, how='inner')

        # Add permit location as a column
        merged = merged.merge(
            permits[['index_permits', 'geometry']].rename(columns={'geometry': 'permit_geometry'}),
            on='index_permits',
        )
        merged['distance'] = gpd.GeoSeries(merged['geometry']).distance(gpd.GeoSeries(merged['permit_geometry']))

        # Now, for each permit, only keep the closest site it matched to.
        merged = merged.sort_values(['index_permits', 'distance']).drop_duplicates(['index_permits'], keep='first').reset_index(drop=True)

        merged = merged.drop(columns=['permit_geometry', 'distance', 'index_sites', 'index_right', 'index_permits'])

        return merged

    else:
        # Buffer only 1 meter, to avoid false positives
        permits.geometry = permits.geometry.buffer(1)

        return gpd.sjoin(sites, permits, how='left')


def geocode_result_to_point(geocodio_result: dict) -> Optional[Point]:
    results = geocodio_result['results']
    if len(results) == 0:
        return None
    # There might be 0 results, or 2 results (one from City of San Jose, one from Santa Clara County for example).
    # Fuck it, just assume the first one is correct
    location = results[0]['location']
    return Point(location['lng'], location['lat'])


def geocode_results_to_geoseries(results: List[dict], index: Optional[pd.Index] = None) -> gpd.GeoSeries:
    return gpd.GeoSeries(
        map(geocode_result_to_point, results),
        index=index,
        crs='EPSG:4326'
    )

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
    ax.set_title(f' {title}', fontdict={'fontsize': 25})
    file_name_prefix = file_name_prefix.replace('/', '')
    file_name_prefix = file_name_prefix.replace(' ', '_')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, attribution=False)
    plt.savefig(f'figures/{file_name_prefix.lower()}_bay_map.jpg')


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
