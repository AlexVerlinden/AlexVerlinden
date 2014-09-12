# calculating distances from samples to roads
require (maptools)
require(rgeos)
library (rgdal)
trial_pts= read.csv("/Users/alexverlinden/grassdata/GIS_Kenya/Kenya_trial_sites/Kenya_noduplicates.csv")
#promote to spatial points frame
coordinates (trial_pts)= ~x+y
#setting projection 
proj4string(trial_pts)= CRS("+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0") # reproject vector file to LAEA
kenya_cp= readShapeSpatial("/Users/alexverlinden/Documents/OFRA/datasets/Kenya Legacy data/KE_cultivated_2003_FAO_highlands.shp", proj4string=CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")) #get shapefile into R and set projection
kenya_crops=spTransform(kenya_cp, CRS("+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")) # reproject vector file to LAEA
kcp=spsample(kenya_crops, n=200, type= 'random') #creates SpatialPoints
kcp2=as.data.frame(kcp)
coords=coordinates(kcp2)
kcp3=SpatialPointsDataFrame(coords, kcp2)
proj4string(kcp3)=CRS("+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")
kenya_roads=readShapeSpatial("/Users/alexverlinden/grassdata/GIS_Kenya/Kenya_roads/kenya_roads.shp")
# set projection
proj4string(kenya_roads)=CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")
#transform to LAEA
kenya_rds_laea=spTransform(kenya_roads,CRS("+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")) # reproject vector file to LAEA
#calculate distance to nearest road for maize trials
#loop for calculating distance by sample
shortest.dists <- numeric(nrow(trial_pts)) # creates empty row of trial obs
for (i in seq_len(nrow(trial_pts)))
{
shortest.dists[i] <- gDistance(trial_pts[i,], kenya_rds_laea)
}
#loop for calculating distances to roads from random points
random.dists <- numeric(nrow(kcp3)) # creates empty row of trial obs
for (i in seq_len(nrow(kcp3)))
{
  random.dists[i] <- gDistance(kcp3[i,], kenya_rds_laea)
}
#ESDA on distances
boxplot(shortest.dists, random.dists, main= "Trials and random samples' distance to main road", ylab= "distance (m)", xlab= "trial samples            random samples")
#t test on means
t.test(shortest.dists, random.dists)
