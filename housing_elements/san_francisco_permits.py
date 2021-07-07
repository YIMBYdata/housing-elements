import pandas as pd
import geopandas as gpd
from shapely.wkt import loads

def load_all_permits(filter_post_2015_new_construction: bool = True, dedupe: bool = True) -> pd.DataFrame:
    permits = pd.read_csv('https://data.sfgov.org/api/views/p4e4-a5a7/rows.csv?accessType=DOWNLOAD')
    date_cols = [c for c in permits.columns if 'Date' in c]
    permits[date_cols] = permits[date_cols].apply(pd.to_datetime)
    permits['apn'] = permits['Block'] + permits['Lot']
    permits['new_units'] = permits['Proposed Units'].fillna(0) - permits['Existing Units'].fillna(0)
    relevant_uses = [
        'apartments', '1 family dwelling', '2 family dwelling',
        'residential hotel', 'misc group residns.', 'artist live/work',
        'convalescent home', 'accessory cottage', 'nursing home non amb',
        'orphanage', 'r-3(dwg) nursing', 'nursing home gt 6'
    ]
    rhna_permits = permits[
        permits['new_units'] > 0
        & permits['Proposed Use'].isin(relevant_uses)
        & permits['Permit Type'].isin([1, 2, 3, 8])
    ]

    if dedupe:
        rhna_permits.sort_values(by="Permit Type", axis=0, inplace=True)
        rhna_permits = rhna_permits.drop_duplicates(['apn', 'new_units', 'Proposed Units', 'Street Number'])

    if filter_post_2015_new_construction:
        rhna_permits = rhna_permits[
            (rhna_permits['Issued Date'] >= '2015-01-01')
        ]

    # Add / rename columns to fit ABAG format
    rhna_permits['permyear'] = rhna_permits['Issued Date'].dt.year
    rhna_permits = rhna_permits.rename({'new_units': 'totalunit'}, axis=1)

    # Address is split up into multiple columns. Must re-combine.
    id_on_street = rhna_permits['Street Number'].astype(str) + " " + rhna_permits['Street Number Suffix'].fillna("")
    street = rhna_permits['Street Name'] + ' ' + rhna_permits['Street Suffix']
    rhna_permits['address'] = id_on_street + ' ' + street

    rhna_permits = rhna_permits.rename(columns={'Location': 'geometry'})

    # Parse string representation of geometry (e.g. 'POINT (-122.4124068139322 37.78615820417518)') into an actual Point object.
    # Not sure why it's in this format in the first place...
    rhna_permits['geometry'] = rhna_permits['geometry'].dropna().apply(loads)

    digit_apns = rhna_permits['apn'].str.isdigit()
    rhna_permits['apn'] = rhna_permits['apn'][digit_apns].astype(int).astype('Int64').reindex(rhna_permits.index, fill_value=pd.NA)

    return gpd.GeoDataFrame(rhna_permits, crs='EPSG:3857')
