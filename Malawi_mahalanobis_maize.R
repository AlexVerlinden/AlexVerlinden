#Maize crop distribution of Malawi based on Maize fertilizer trials
#conducted during 1996-1998 

#Presence only  model no absence

## Malawi LREP response trial data (courtesy of LREP & Todd Benson)
# LREP data documentation at: https://www.dropbox.com/s/4qbxnz4mdl92pdv/Malawi%20area-specific%20fertilizer%20recs%20report.pdf?dl=0
#Alex Verlinden, January 2015
#presence only data except for Malawi other projects have currently few points

# load Required packages 


#install.packages(c("downloader","proj4","dismo", "maptools", "rgdal","raster","rgeos"))

require(downloader)
require(proj4)
require(dismo)
require(maptools)
require(rgdal)
require(raster)
require(rgeos)


# Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("Data", showWarnings=F)
dat_dir <- "./Data"
# LREP fertilizer response data download to "./Data"
download("https://www.dropbox.com/s/rra8c3gcx8bjjnn/MW_fert_trials.zip?dl=0","./Data/MW_fert_trials.zip", mode="wb")

unzip("./Data/MW_fert_trials.zip", exdir="./Data", overwrite=TRUE)
mwsite <- read.table(paste(dat_dir, "/Location.csv", sep=""), header=TRUE, sep=",")
mtrial <- read.table(paste(dat_dir, "/Trial.csv", sep=""), header=TRUE, sep=",")

# Georeference and specify site ID's --------------------------------------
# Project to Africa LAEA from UTM36S
mw <- cbind(mwsite$Easting, mwsite$Northing)
tr <- ptransform(mw, '+proj=utm +zone=36 +south +datum=WGS84 +units=m +no_defs', '+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs')
colnames(tr) <- c("x","y","z")
mwsite <- cbind(mwsite, tr)

# Define unique grid cell / site ID's (GID)
# Specify pixel scale (res.pixel, in m)
res.pixel <- 1000

# Grid ID (GID) definition
xgid <- ceiling(abs(mwsite$x)/res.pixel)
ygid <- ceiling(abs(mwsite$y)/res.pixel)
gidx <- ifelse(mwsite$x<0, paste("W", xgid, sep=""), paste("E", xgid, sep=""))
gidy <- ifelse(mwsite$y<0, paste("S", ygid, sep=""), paste("N", ygid, sep=""))
GID <- paste(gidx, gidy, sep="-")
mwsite.gid <- cbind(mwsite, GID)

#load Malawi boundary file (laea projection)
download("https://www.dropbox.com/s/emp44mx75je6g08/MW_admin0.zip?dl=0", "./Data/MW_adm0.zip", mode="wb")
#unzip
unzip("./Data/MW_adm0.zip", exdir="./Data", junkpaths= TRUE, overwrite=TRUE)
mw_bound=readShapeSpatial(paste(dat_dir, "/MW_admin0_laea.shp",sep=""))

# Malawi grids download to "./Data" (~7.6 Mb)
download("https://www.dropbox.com/s/gqk6y13bvui3egk/MW_grids.zip?dl=0", "./Data/MW_grids.zip", mode="wb")
#unzip
unzip("./Data/MW_grids.zip", exdir="./Data", junkpaths=TRUE, overwrite=TRUE)
glist <- list.files(path="./Data", pattern="tif", full.names=TRUE)
mwgrid <- stack(glist)
# Malawi maize presence data
mwsite_locs=mwsite.gid[,4:5]
#extract covariates at presence sites
mp=extract(grids, mwsite_locs)
mpvar=cbind(mwsite.gid, mp)
# background random data within 10 km buffer around presence data
download("https://www.dropbox.com/s/7gjh5tungoyr7os/MW_random_laea.zip?dl=0", "./Data/MW_random_laea.zip", mode="wb")
#unzip
unzip("./Data/MW_random_laea.zip", exdir="./Data", junkpaths=TRUE, overwrite=TRUE)
mw=readShapeSpatial(paste(dat_dir, "/MW_random_laea.shp", sep=""))
#extract covariates at random background data within 10km from trials
mwback=extract(grids, mw)
#get xy coordinates for background data from data frame
mw_loc=as.data.frame(mw)
mw_loc=mw_loc[,1:2]
#put coordinates and variables together
mwvar=cbind(mw_loc, mwback)
#put presence and background data together
mw_pb=c(rep(1, nrow(mwsite.gid)), rep(0, nrow(mw_loc)))
mw_presback=data.frame(cbind(mw_pb, rbind(mp, mwback)))
#divide in training and test data Malawi
mwsamp=sample(nrow(mpvar), round(0.75*nrow(mpvar)))
traindata=mpvar[mwsamp,]
train=traindata[,1:2]
traindata=traindata[,3:17]
testdata=mwvar[-mwsamp,]
test=testdata[,1:2]
testdata=testdata[,3:17]

#use k-fold partitioning on train data
pres=mw_presback[mw_presback[,1]==1,2:16]
back=mw_presback[mw_presback[,1]==0, 2:16]
k=5
group=kfold(pres,k)
group[1:10]
unique(group)
e=list()
for (i in 1:k) {
  train =pres[group !=i,]
  test= pres[group==i,]
  mh=mahal(train)
  e[[i]] = evaluate(p=test, a=back, mh)
}
auc <- sapply( e, function(x){slot(x, 'auc')} )
auc
mean (auc)
#maximum of the sum of the sensitivity (true positive rate) and specificity (true negative rate)
sapply( e, function(x){ x@t[which.max(x@TPR + x@TNR)] } )

#test if removing spatial sorting bias Malawi is needed
nr <- nrow(mwsite_locs)
seed=12345
s <- sample(nr, 0.75 * nr)
pres_train <- mwsite_locs[-s, ]
pres_test <- mwsite_locs[s, ]
nr <- nrow(mw_loc)
s <- sample(nr, 0.75 * nr)
back_train <- mw_loc[-s, ]
back_test <- mw_loc[s, ]
sb <- ssb(pres_test, back_test, pres_train)
sb[,1] / sb[,2] #if p is close to 1, sorting bias is very low

#mahalanobis evaluation of AUC
#mh1=mahal(grids, pres_train)
#e=evaluate(pres_test, back_test, mh1, grids)
#e
#for sample trial representativeness an AUC close to 0.5 is preferred, showing experiments are representative for neighbouring environment
plot(e, 'ROC')
#prediction using the grids rasterstack for malawi
#training data only
pm=predict(grids,mh, progress= '') #using mahal for all training data
#setting threshold for minimum in plot
tr=threshold(e, 'spec_sens')
pm2=pm
pm2[pm2<=tr]=tr
plot (pm2)
points(mwsite_locs, cex=0.01, col='blue')
plot(mw_bound, add=TRUE)
rf=writeRaster(pm2, filename="Mw_mahalanobis", format= "GTiff", overwrite=TRUE)
#all data
mh_all=mahal(grids, mwsite_locs)
pm_all=predict(grids,mh_all, progress='')
pm_all[pm_all<=-10]=-10
plot(pm_all)
#write to Tiff file

rf=writeRaster(pm_all, filename="Mw_mahalanobis_all", format= "GTiff",overwrite= TRUE)