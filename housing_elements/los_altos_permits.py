import pandas as pd
import geopandas as gpd
import os


def load_all_permits(filter_post_2015_new_construction: bool = True, dedupe: bool = True) -> pd.DataFrame:
    """
    TODO:
    la_sites.apn = la_sites.apn.str.replace('-','').astype('float')
    sites APN is in different format. Handle in utils.py
    """
    path = os.path.join(os.path.dirname(__file__), '../data/clean_data/los_altos_permits.csv')
    permits = pd.read_csv(path)
    permits.dropna(how='all', axis='columns', inplace=True)
    date_cols = ['issued', 'final', 'applied']
    permits[date_cols] = permits[date_cols].apply(pd.to_datetime)
    renaming_map = {
        'APN': 'apn',
        'RHNA.Date': 'permyear',
        'Site.Address': 'address',
        'net': 'totalunit'
    }
    permits = permits.rename(renaming_map, axis=1)

    if filter_post_2015_new_construction:
        permits = permits.query('permyear >= 2015 and permyear < 2020')

    if dedupe:
        permits = permits.sort_values('totalunit', ascending=False).drop_duplicates('apn')

    # Convert float to Int64 (nullable int)
    permits['apn'] = permits['apn'].dropna().astype('int64').astype('Int64').reindex(permits.index, fill_value=pd.NA)

    return permits
