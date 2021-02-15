# Housing Elements analysis

The goal here (for now) is to figure out what the correct "realistic capacity" should be for sites that San Francisco (and maybe other cities) intends to put in its Sixth RHNA Cycle Housing Elements.

HCD's recent [memo](https://www.hcd.ca.gov/community-development/housing-element/docs/sites_inventory_memo_final06102020.pdf) on AB1397 explains that sites must be discounted to address the fact that not all will be built, and built to the maximum zoned density, within the next eight years. For example, if San Francisco has a RHNA of 80,000 and only about 1 in 10 sites are expected to be developed in eight years, then it should zone for 800,000 units.

## Data overview
The Bay Area's housing sites from the 5th cycle (2015-2023) are available at https://opendata.mtc.ca.gov/datasets/da0765ab82ae475d985688e140f931bd_0/data?geometry=-130.241%2C36.376%2C-114.431%2C39.410. We can then join this to the San Francisco's building permits [dataset](https://data.sfgov.org/Housing-and-Buildings/Building-Permits/i98e-djp9) to figure out how many of the sites actually applied for a building permit, how many received a permit, and how many actually started or finished construction. We can also compare the predicted density in the housing element (in the MTC dataset) to the number of units actually built.

## Setting up the environment
This repo uses [`poetry`](https://python-poetry.org/) to set up the dependencies/virtualenv. After installing poetry, run
```sh
poetry install
poetry run jupyter lab
```
to spin up a Jupyter lab shell with all of the dependencies.

## Getting the data
To be able to run the notebook, you will need to run the following:
```sh
wget https://opendata.arcgis.com/datasets/da0765ab82ae475d985688e140f931bd_0.zip?outSR=%7B%22latestWkid%22%3A4326%2C%22wkid%22%3A4326%7D -O housing_sites.zip
mkdir data/raw_data/housing_sites
unzip housing_sites.zip -o data/raw_data/housing_sites
```

The SF permits data is included in data/raw_data/sf_permits.csv, and the data is up-to-date as of 2/15/2021. If you would like to retrieve permits issued after 2/15/2021, you can run this code chunk:

```py
sf_permits = pd.read_csv('https://data.sfgov.org/api/views/p4e4-a5a7/rows.csv?accessType=DOWNLOAD')
sf_permits.to_csv('./data/raw_data/sf_permits.csv', index=False)
```