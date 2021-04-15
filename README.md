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

## Data sources
Documentation on how the data was obtained are in [Data sources](<Data sources.md>).
