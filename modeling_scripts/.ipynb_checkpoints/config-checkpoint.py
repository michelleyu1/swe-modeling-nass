# Externals
import pandas as pd
import geopandas as gpd
from shapely import wkt
from datetime import datetime
from sklearn.model_selection import train_test_split

# Locals
import misc



### Define global variables ###

# Study period
start_year = datetime(1985,10,1)
end_year = datetime(2016,9,30)

# Features
full_features = ['pr', 'tmin', 'tmax', 'temp', 'swe', 'swe_t1', 'dah', 'trasp', 'elev', 'elevGTOPO', 'slopeGTOPO', 'aspectGTOPO', 'tmin_t1', 'tmax_t1', 'temp_t1', 'tmin_t3', 'tmax_t3', 'temp_t3', 'tmin_t7', 'tmax_t7', 'temp_t7']
features = ['pr', 'temp', 'tmin', 'tmax', 'dah', 'trasp', 'elev', 'slopeGTOPO', 'aspectGTOPO', 'pr_t7', 'temp_t7', 'tmin_t7', 'tmax_t7']    

# Model configuration
model_type = 'dswe'
output = 'dswe'

# Define relevant directories
data_dir = '../data/'
save_dir = f'../results/{model_type}/'   
uaswe_dir = data_dir+'UofASWE/' 

### Get SNOTEL sites in UCRB study area ###
# Load list of SNOTEL sites with available data
sites_df = pd.read_csv(data_dir+'wus_sntl_sites_df.csv', index_col='Unnamed: 0')
sites_df['geometry'] = sites_df['geometry'].apply(wkt.loads)
# Convert to a Geopandas gdf
sites_gdf = gpd.GeoDataFrame(sites_df, crs='EPSG:4326')
# Get shapefile for Upper Colorado Riber Basin (UCRB)
uc_shp = "../data/Upper_Colorado_River_Basin_Boundary/Upper_Colorado_River_Basin_Boundary.shp"
# Read UCRB shapefile
gm_poly_gdf = gpd.read_file(uc_shp, encoding="utf-8")
# Get bounds of UCRB
gm_poly_geom = gm_poly_gdf.iloc[0].geometry
# Determine sites in UCRB
sites_idx = sites_gdf.intersects(gm_poly_geom)
# Subset df to sites in UCRB
gm_snotel_sites = sites_gdf.loc[sites_idx]

# Load pre-processed dataframe with attribute data for all site-years
all_sites_df = pd.read_csv(data_dir+'all_sites_df.csv', index_col=0)


### Train-test split of water years ###
trn, tst = train_test_split(list(range(start_year.year+1, end_year.year+1)), test_size=0.2, random_state=27106)
# Cross validation split
cv_splits = misc.partition(trn, 4)

