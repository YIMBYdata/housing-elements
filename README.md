# Housing Elements analysis

The goal here (for now) is to figure out what the correct "realistic capacity" should be for sites in various California cities' Sixth RHNA Cycle Housing Elements.

HCD's recent [memo](https://www.hcd.ca.gov/community-development/housing-element/docs/sites_inventory_memo_final06102020.pdf) on AB1397 explains that sites must be discounted to address the fact that not all will be built within the next eight years. For example, if San Francisco has a RHNA of 80,000 and only about 1 in 10 sites are expected to be developed in eight years, then it should zone for 800,000 units.

## Data overview
The Bay Area's housing sites from the 5th cycle (2015-2023) are available at the [MTC OpenData portal](https://opendata.mtc.ca.gov/datasets/da0765ab82ae475d985688e140f931bd_0/data?geometry=-130.241%2C36.376%2C-114.431%2C39.410). We can then join this to various city data sources to figure out how many of the sites actually applied for a building permit, how many received a permit, and how many actually started or finished construction. We can also compare the predicted density in the housing element (in the MTC dataset) to the number of units actually built.

We also used San Francisco's [parcels shapefile](https://data.sfgov.org/Geographic-Locations-and-Boundaries/Parcels-Active-and-Retired/acdm-wktn) to figure out the lot areas for various building permits.

## Setting up the environment
This repo uses [`poetry`](https://python-poetry.org/) to set up the dependencies/virtualenv. After installing poetry, run
```sh
poetry install
poetry run jupyter lab
```
to spin up a Jupyter lab shell with all of the dependencies.

Additionally, you must create a geocodio API key at geocod.io, and in the project root create a file "geocodio_api_key.json" with the contents:
    {
        "key": "INSERT_YOUR_API_KEY_HERE"
    }

## Data sources
Documentation on how the data was obtained are in [Data sources](<Data sources.md>).

## Data format
We got our raw data from two sources:
* ABAG (from Annual Progress Reports)
* City sources

The helpers module `housing_elements/utils.py` provides a function `load_all_new_building_permits` that returns all building permits from 2013 to 2019 for most ABAG jursidictions.

We also have helper functions that provide datasets in the same format, pulled from the city's online data portals. We used both sources to confirm that the ABAG dataset is valid. We have helpers for the following cities:
* San Jose (`load_new_construction_permits` in `housing_elements/san_jose_permits.py`)
* San Francisco (`load_new_construction_permits` in `housing_elements/san_francisco_permits.py`)
* Los Altos (`load_new_construction_permits` in `housing_elements/los_altos_permits.py`)

All of these return a DataFrame that have at minimum the following columns:
* _permyear_ (int): Year that the project was permitted
* _apn_ (str). APNs **are allowed to be duplicated**, if there are multiple projects (with different addresses) happening on the same parcel. The consumer of this dataset is expected to group by APN and sum up `totalunit` to get the total number of units built on that parcel.
* _address_ (str). like `apn`, also allowed to be duplicated if they're actually separate building projects on the same APN.
* _totalunit_ (int): Number of new units built


