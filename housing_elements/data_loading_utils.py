from __future__ import annotations
import re
import os
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from pandas.api.types import is_numeric_dtype
from housing_elements import geocode_cache
from shapely.geometry import Point
from typing import List, Optional

_logger = logging.getLogger(__name__)

ABAG = None
INVENTORY = None
TARGETS = None
BPS_DATA = None
XLSX_FILES = [
    ('Richmond', '2018'),
    ('PleasantHill', '2018'),
    ('Oakland', '2019'),
    ('Livermore', '2019'),
]
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

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

    # Filter out permits from before the start of the 5th Housing Element cycle.
    ABAG = ABAG[ABAG['permyear'] >= 2015].copy()

    ABAG['apn'] = ABAG['apn'].replace({np.nan: None})

    return ABAG


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

    permits_df['apn_raw'] = permits_df['apn']
    permits_df['apn'] = standardize_apn_format(permits_df['apn'])

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

SITE_TYPES_TO_EXCLUDE = [
    'Approved',
    'Built',
    'Entitled',
    'Planned and Approved',
    'Under Construction',
]


VACANT_SITE_TYPES = ['Underutilized and Va', 'Vacant', 'Undeveloped', 'Open Space', 'Underutilized & Vaca', 'Vacant and Underutil']

NONVACANT_SITE_TYPES = ['Opportunity', 'Underused site', 'Underutilized, margi', 'Non-Vacant', "Infill", 'underutilize', 'Underutilized']

def load_site_inventory(city: str, exclude_approved_sites: bool = True, fix_realcap: bool = True) -> pd.DataFrame:
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

    if exclude_approved_sites:
        # Keep sites where sitetype is null or sitetype != Approved.
        # I guess it's possible that some null rows are also pre-approved, but whatever. We can
        # document that as a potential data issue.
        rows_to_keep &= (~sites_df['sitetype'].isin(SITE_TYPES_TO_EXCLUDE)).fillna(True)

    sites = sites_df[rows_to_keep].copy()
    sites.fillna(value=np.nan, inplace=True)

    if fix_realcap:
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

    # Back up the raw apn/locapn, so that we can calculate the number of matches using the raw string.
    sites['apn_raw'] = sites['apn'] # float_col_to_nullable_int(pd.to_numeric(sites['apn'], errors='coerce'))
    sites['locapn_raw'] = sites['locapn'] # float_col_to_nullable_int(pd.to_numeric(sites['locapn'], errors='coerce'))

    sites['apn'] = standardize_apn_format(sites['apn'])
    sites['locapn'] = standardize_apn_format(sites['locapn'])

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
    sites['is_vacant'] = sites.sitetype.isin(VACANT_SITE_TYPES)
    sites['is_nonvacant'] = sites.sitetype.isin(NONVACANT_SITE_TYPES)
    sites['na_vacant'] = ~sites.sitetype.isin(VACANT_SITE_TYPES + NONVACANT_SITE_TYPES)
    return sites

def float_col_to_nullable_int(series: pd.Series) -> pd.Series:
    """
    Given a pd.Series with float values (possibly null),
    rounds the non-nan values to their nearest int, and fills the null values in with None.

    Returns a Series of dtype 'Int64' (the new nullable int type).
    """
    return series.dropna().astype('int64').astype('Int64').reindex(series.index, fill_value=pd.NA)

def standardize_apn_format(column: pd.Series) -> pd.Series:
    if not is_numeric_dtype(column.dtype):
        column = column.str.replace("-", "", regex=False)
        column = column.str.replace(" ", "", regex=False)
        column = column.str.replace(r"[a-zA-Z|.+,;:/]", '', regex=True)

        # Some extra logic to handle Oakland, Hayward, Pittsburg, San Bruno, South SF, Windsor
        column = column.str.replace("\n", "", regex=False)
        column = column.where(column.str.isdigit())
        column = column.where(column.str.len() > 0)
        column = column.where(column.str.len() <= 23)

        column = pd.to_numeric(column, errors='coerce')

    column = float_col_to_nullable_int(column)
    return column

def drop_constant_cols(sites: pd.DataFrame) -> pd.DataFrame:
    """Return df with constant columns dropped unless theyre necessary for QOI calculations."""
    if len(sites.index) > 1:
        dont_drop = ['existuse', 'totalunit', 'permyear', 'relcapcty', 'apn', 'sitetype',
                     'realcap_parse_fail', 'realcap_not_listed']
        is_constant = ((sites == sites.iloc[0]).all())
        constant_cols = is_constant[is_constant].index.values
        constant_cols = list(set(constant_cols) - set(dont_drop))
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

def get_rhna_target(city: str) -> float:
    rhna_targets = load_rhna_targets()
    if city == 'Overall':
        return rhna_targets['Total'].sum()
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        rhna_target = rhna_targets.query('City == @city')['Total'].values[0]
    return rhna_target

def get_census_bps_dataset() -> pd.DataFrame:
    global BPS_DATA
    if BPS_DATA is None:
        # Census building permits dataset for places (i.e. cities or unincorporated places)
        # compiled by Sid for another project.
        # Downloaded from https://housingdata.app/places_annual.parquet.
        BPS_DATA = pd.read_parquet(
            'data/raw_data/places_annual.parquet',
            columns=['year', 'place_name', 'state_code', 'total_units']
        )
        BPS_DATA = BPS_DATA[
            (BPS_DATA['year'] >= '2015')
            & (BPS_DATA['year'] <= '2019')
            & (BPS_DATA['state_code'] == 6)  # FIPS code for California
        ].drop(columns=['state_code']).copy()

        # That dataset accidentally drops the ' City' in a lot of cities whose names end in 'City'
        cities_to_fix = ['Foster', 'Union', 'Redwood', 'Daly', 'Suisun']
        for city in cities_to_fix:
            BPS_DATA.loc[BPS_DATA['place_name'] == city, 'place_name'] = city + ' City'

    return BPS_DATA

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
