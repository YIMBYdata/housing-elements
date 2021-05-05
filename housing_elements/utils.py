import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from shapely.geometry import Point
from typing import List, Optional
from . import geocode_cache


_logger = logging.getLogger(__name__)

XLSX_FILES = [
    ('Richmond', '2018'),
    ('Pleasant Hill', '2018'),
    ('Oakland', '2019'),
    ('Livermore', '2019'),
]

def load_apr_permits(
    city: str, year: str, filter_for_permits: bool = True
) -> pd.DataFrame:
    """
    :param year: must be either '2018' or '2019'
    """
    city = city.replace(" ", "")

    if (city, year) in XLSX_FILES:
        path = f"data/raw_data/APRs/{city}{year}.xlsx"
    else:
        path = f"data/raw_data/APRs/{city}{year}.xlsm"

    df = pd.read_excel(
        path,
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
    geometry_df = gpd.read_file("data/raw_data/abag_building_permits/permits.shp")
    data_df = pd.read_csv("data/raw_data/abag_building_permits/permits.csv")

    # There shouldn't be any rows with geometry data that don't have label data
    assert geometry_df["joinid"].isin(data_df["joinid"]).all()

    return gpd.GeoDataFrame(data_df.merge(geometry_df, how="left", on="joinid"))


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
    missing_indices = df[df.geometry.isnull() & df['address'].notnull()].index

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
    abag_permits_df = abag_permits_df[abag_permits_df["jurisdictn"] == city]

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

    # We need to add "<city name>, CA" to the addresses when we're geocoding them because the ABAG dataset (as far as I've seen)
    # doesn't have the city name or zip code. Otherwise, we get a bunch of results of that address from all over the US.
    return impute_missing_geometries(permits_df, address_suffix=f', {city}, CA')

def load_site_inventory(city: str) -> pd.DataFrame:
    """
    Return the 5th RHNA cycle site inventory for CITY.
    """
    df = gpd.read_file(
        "./data/raw_data/housing_sites/xn--Bay_Area_Housing_Opportunity_Sites_Inventory__20072023_-it38a.shp"
    )
    assert (
        city in df.jurisdict.values
    ), "city must be a jurisdiction in the inventory. Be sure to capitalize."
    sites = df.query(f'jurisdict == "{city}" and rhnacyc == "RHNA5"').copy()
    sites.fillna(value=np.nan, inplace=True)

    # MV & PA uses a range for some values. Following line replaces range with max.
    sites.allowden = sites.allowden.str.replace('du/ac','')
    sites.allowden = sites.allowden.str.split('-').str[-1]
    sites['allowden'] = sites['allowden'].astype(float)
    sites['relcapcty'] = sites['relcapcty'].astype(float)
    is_constant = ((sites == sites.iloc[0]).all())

    constant_cols = is_constant[is_constant].index.values
    print("Dropping constant columns:", constant_cols)
    sites.drop(constant_cols, axis=1, inplace=True)
    sites.dropna(how="all", axis=1, inplace=True)
    sites.apn = sites.apn.str.replace("-", "")
    print("DF shape", sites.shape)
    return sites


def calculate_inventory_housing_over_all_housing(
    sites: pd.DataFrame, permits: pd.DataFrame
) -> float:
    """(new housing units on HE sites) / (new housing units)"""
    housing_on_sites = permits[permits.apn.isin(sites.apn)].totalunit.sum()
    total_units = permits.totalunit.sum()

    print("Units permitted on inventory sites:", housing_on_sites)
    print("Total units permitted:", total_units)

    return housing_on_sites / total_units


def calculate_mean_overproduction_on_sites(
    sites: pd.DataFrame, permits: pd.DataFrame
) -> float:
    """mean(housing units - HE claimed capacity), with mean taken over HE sites that were developed"""
    inventory_sites_permitted = permits[permits.apn.isin(sites.apn)].apn.unique()
    n_units = permits[permits.apn.isin(inventory_sites_permitted)].totalunit.sum()
    n_claimed = sites[sites.apn.isin(inventory_sites_permitted)].relcapcty.sum()
    print("Number of inventory sites developed:", len(inventory_sites_permitted))
    print("Number of units permitted on inventory sites:", n_units)
    print("Total realistic capacity of inventory sites:", n_claimed)
    return (n_units - n_claimed) / len(inventory_sites_permitted)


def calculate_total_units_permitted_over_he_capacity(sites: pd.DataFrame, permits: pd.DataFrame) -> float:
    """ (total units permitted) / (HE site capacity)
    """
    total_units = permits.totalunit.sum()
    total_inventory_capacity = sites.relcapcty.sum()
    print("Total units permitted:", total_units)
    print("Total realistic capacity in inventory:", total_inventory_capacity)
    return total_units / total_inventory_capacity


def calculate_pdev_for_inventory(sites: pd.DataFrame, permits: pd.DataFrame) -> float:
    """Return P(permit | inventory_site)"""
    return sites.apn.isin(permits.apn).mean()


def calculate_pdev_for_vacant_sites(sites: pd.DataFrame, permits: pd.DataFrame) -> float:
    """Return P(permit | inventory_site, vacant)"""
    is_permitted = sites.apn.isin(permits.apn)
    is_vacant = sites.sitetype == 'Vacant'
    n_vacant = is_vacant.sum()
    n_vacant_permitted = (is_permitted & is_vacant).sum()
    return n_vacant_permitted / n_vacant


def calculate_pdev_for_nonvacant_sites(sites: pd.DataFrame, permits: pd.DataFrame):
    """Return P(permit | inventory_site, non-vacant)"""
    is_permitted = sites.apn.isin(permits.apn)
    is_nonvacant = sites.sitetype != 'Vacant'
    n_nonvacant = is_nonvacant.sum()
    n_nonvacant_permitted = (is_permitted & is_nonvacant).sum()
    return n_nonvacant_permitted / n_nonvacant


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
