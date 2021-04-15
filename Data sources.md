# Data sources
Here's a list of each of the data sources we used.
There are three sources of data: the Bay Area site inventory, the parcels dataset, and the SF permits dataset. This data is included in data/raw_data. 

* Bay Area 5th cycle site inventory dataset

**Source:** https://opendata.mtc.ca.gov/datasets/da0765ab82ae475d985688e140f931bd_0

**Location:** `data/raw_data/housing_sites/xn--Bay_Area_Housing_Opportunity_Sites_Inventory__20072023_-it38a.shp`

**Download script:**
```sh
wget https://opendata.arcgis.com/datasets/da0765ab82ae475d985688e140f931bd_0.zip?outSR=%7B%22latestWkid%22%3A4326%2C%22wkid%22%3A4326%7D -O housing_sites.zip
mkdir data/raw_data/housing_sites
unzip housing_sites.zip -o data/raw_data/housing_sites
``

* San Francisco parcels

**Download script:**
```bash
wget 'https://data.sfgov.org/api/geospatial/acdm-wktn?method=export&format=Shapefile' -O all_parcels.zip
mkdir data/raw_data/all_parcels
unzip all_parcels.zip -o data/raw_data/all_parcels

# When you unzip the folder, the files will be called `geo_export_{some random string}.{dbf,prj,shp,shx}`.
# The random string will be different every time, so here we rename the files to `all_parcels.{dbf,prj,shp,shx}`.
# so that the notebooks work regardless of what the downloaded files are called.
for f in $(ls data/raw_data/all_parcels)
do 
    mv data/raw_data/all_parcels/$f data/raw_data/all_percels/all_parcels.${f##*.}
done
``**

* San Francisco permits
**Download script:**
```py
import pandas as pd
sf_permits = pd.read_csv('https://data.sfgov.org/api/views/p4e4-a5a7/rows.csv?accessType=DOWNLOAD')
sf_permits.to_csv('./data/raw_data/sf_permits.csv', index=False)
```

For reproducibility's sake, these notebooks stick to the SF permits data as of 2/15/2021. If you would like to update this data and retrieve permits issued after 2/15/2021, just run the code chunk above.
