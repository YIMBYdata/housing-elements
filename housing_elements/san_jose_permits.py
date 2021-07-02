import pandas as pd
import geopandas as gpd


def load_all_permits(filter_post_2015_new_construction: bool = True, dedupe: bool = True) -> pd.DataFrame:
    expired_permits = gpd.read_file('data/raw_data/san_jose/sj_expired_building_permits.shp')
    active_permits = gpd.read_file('data/raw_data/san_jose/sj_active_building_permits.shp')

    expired_permits.dropna(how='all', axis='columns', inplace=True)
    active_permits.dropna(how='all', axis='columns', inplace=True)

    permits_df = pd.concat(
        [
            expired_permits.assign(state='expired'),
            active_permits.assign(state='active'),
        ],
        ignore_index=True,
    )
    permits_df = permits_df.to_crs('EPSG:4326')

    permits_df['ISSUEDATE'] = pd.to_datetime(permits_df['ISSUEDATE'])
    permits_df['ISSUEDATEU'] = pd.to_datetime(permits_df['ISSUEDATEU'])
    permits_df['LASTUPDATE'] = pd.to_datetime(permits_df['LASTUPDATE'])

    permits_df = permits_df.rename(
        columns={
            'ADDRESS': 'address',
            'APN': 'apn',
            'DWELLINGUN': 'totalunit',
        }
    )
    # Reorder columns for convenience
    permits_df = permits_df[
        ['apn', 'address', 'totalunit'] + list(set(permits_df.columns) - {'apn', 'address', 'totalunit'})
    ]

    permits_df['permyear'] = permits_df['ISSUEDATE'].dt.year

    if filter_post_2015_new_construction:
        permits_df = permits_df[permits_df['permyear'] >= 2015]

        filtered_permits_df = permits_df[(permits_df['WORKDESC'] == "New Construction") & (permits_df['totalunit'] > 0)]

        # There are many rows with a data error, where new units = 0 even though there is actually a new unit added.
        # This should catch some of them...
        typo_permits_df = permits_df[
            (permits_df['WORKDESC'] == "New Construction")
            & (permits_df['totalunit'] == 0)
            & permits_df['SUBDESC'].isin(['2nd Unit Added', 'Single-Family'])
        ].copy()
        typo_permits_df['corrected'] = True
        typo_permits_df['totalunit'] = 1

        permits_df = pd.concat([filtered_permits_df, typo_permits_df])

        assert (
            permits_df['SUBDESC']
            .isin(
                [
                    '2nd Unit Added',
                    'Single-Family',
                    'Apt/Condo/Townhouse',
                    'Mixed Use',
                    'Condo',
                    'Apartment',
                    'Duplex',
                    'Townhouse',
                    'Manufactured Home',
                ]
            )
            .all()
        )

    if dedupe:
        if not filter_post_2015_new_construction:
            raise ValueError("Dedupe without filtering is not an option!")

        # Deduping steps
        dupe_apns = permits_df['apn'].value_counts().loc[lambda x: x > 1].index

        non_dupes = permits_df[~permits_df['apn'].isin(dupe_apns)]
        dupes = permits_df[permits_df['apn'].isin(dupe_apns)]

        deduped_dupes = dupes.groupby('apn').apply(_dedupe_apn).reset_index(drop=True)

        return pd.concat(
            [
                non_dupes,
                deduped_dupes,
            ]
        ).reset_index(drop=True)
    else:
        return permits_df.reset_index(drop=True)


def _dedupe_apn(apn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles a group of rows with the same APN, using heuristics I think are mostly correct based on manually looking at the duplicates.

    We only consider something a "dupe" if it has the same APN and the same address.
    There are only a handful of cases where this is true; looking at them manually, they're mostly correct (actually two different
    buildings at the same address).

    There's only one true "dupe" case (255 E Virginia St), where it has two rows, both representing the same 301 studios project, with different
    total square footages. In that case, we take the last row.
    """
    # These 4 fields should be unique
    assert len(apn_df.drop_duplicates(['apn', 'address', 'SUBDESC', 'SQUAREFOOT'])) == len(apn_df)

    def _process_address_group(address_group):
        if len(address_group) == 1:
            return address_group

        apns = address_group['apn'].unique()
        assert len(apns) == 1
        apn = apns[0]

        if apn in ['47225092', '59519002']:
            # I found that these are actually dupes. return just the last one
            assert len(address_group) == 2
            return address_group.sort_values('ISSUEDATE')[-1:]
        else:
            # These dupes have been manually checked. They're actually separate buildings on the same APN and address.
            assert apn in [
                '25935042',
                '24911077',
                '65445032',
                '25932045',
                '26437060',
                '25905079',
                '67610035',
                '24103020, 24104006,',
            ]
            return address_group

    return apn_df.groupby('address').apply(_process_address_group).reset_index(drop=True)
