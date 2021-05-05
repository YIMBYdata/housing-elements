import pandas as pd
import geopandas as gpd


def load_all_permits(filter_post_2015_new_construction: bool = True, dedupe: bool = True) -> pd.DataFrame:
    sf_building_permits = pd.read_csv("../data/raw_data/sf_permits.csv")
    date_cols = [c for c in sf_building_permits.columns if 'Date' in c]
    sf_building_permits[date_cols] = sf_building_permits[date_cols].apply(pd.to_datetime)
    sf_building_permits['apn'] = sf_building_permits['Block'] + '/' + sf_building_permits['Lot']
    sf_building_permits['new_units'] = sf_building_permits['Proposed Units'].fillna(0) - sf_building_permits['Existing Units'].fillna(0)
    relevant_uses = ['apartments', '1 family dwelling', '2 family dwelling', 
                 'residential hotel', 'misc group residns.', 'artist live/work', 
                 'convalescent home', 'accessory cottage', 'nursing home non amb',
                'orphanage', 'r-3(dwg) nursing', 'nursing home gt 6']
    sf_all_construction = sf_building_permits[
        sf_building_permits['new_units'] > 0
        & sf_building_permits['Proposed Use'].isin(relevant_uses)
        & sf_building_permits['Permit Type'].isin([1, 2, 3, 8])
    ]
    if dedupe:
        sf_all_construction.sort_values(by="Permit Type", axis=0, inplace=True)
        sf_all_construction = sf_all_construction.drop_duplicates(['apn', 'new_units', 'Proposed Units', 'Street Number'])
    if filter_post_2015_new_construction:
        sf_all_construction = sf_all_construction[
            (sf_all_construction['Issued Date'] >= '2015-01-01')
        ]
    return sf_all_construction