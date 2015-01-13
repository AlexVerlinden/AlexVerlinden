# profile analysis of trial sites OFRA
#Maize crop distribution of OFRA locations based on Maize fertilizer trials
#conducted during 2014

#Presence only  model no absence

#Alex Verlinden, January 2015
#presence only data except for Malawi other projects have currently few points
#3 PCA
#Loadings
#                                                  
#i.pca input=som_20_s@PERMANENT,snd_sd2s@PERMANENT,seasons_AfSIS@PERMANENT,phh20s@PERMANENT,gdds_AfSIS@PERMANENT,TMFI@PERMANENT,TMAP@PERMANENT,RELI@PERMANENT,REF7@PERMANENT,REF3@PERMANENT,REF2@PERMANENT,REF1@PERMANENT,LSTN@PERMANENT,LSTD@PERMANENT,EVI@PERMANENT,CTI@PERMANENT,CEC_sd2s@PERMANENT,BSAV@PERMANENT,BSAS@PERMANENT,BSAN@PERMANENT,Ariditys_AfSIS@PERMANENT,ELEV@PERMANENT output_prefix=PC_22_11jan rescale=0,0
#Eigen values, (vectors), and [percent importance]:
#PC1      5.09 ( 0.3244,-0.0907,-0.1268,-0.3707,-0.0282, 0.2133, 0.3239, 0.1889,-0.1856,-0.1423,-0.0504,-0.1538, 0.0032,-0.3745, 0.2869,-0.0869,-0.0713,-0.1555,-0.1372,-0.1260, 0.4078, 0.0698) [38.33%]
#PC2      2.46 (-0.2662, 0.2729,-0.0556,-0.1832, 0.1187, 0.1166, 0.1265,-0.3982,-0.0045, 0.0294, 0.0881, 0.0036, 0.3432, 0.0033, 0.1237, 0.0825,-0.3828, 0.0046, 0.0226, 0.0338, 0.1187,-0.5521) [18.50%]
#PC3      2.00 (-0.1761, 0.5473, 0.1675,-0.1839,-0.1220,-0.0747,-0.0674, 0.0157, 0.0131,-0.0371,-0.0286,-0.0138,-0.4604,-0.1272,-0.0217,-0.0363,-0.4762,-0.0095,-0.0002, 0.0055,-0.0425, 0.3481) [15.05%]
# load Required packages 


#install.packages(c("downloader","proj4","dismo", "maptools", "rgdal","raster","rgeos"))

require(downloader)
require(proj4)
require(dismo)
require(maptools)
require(rgdal)
require(raster)
require(rgeos)

#set working directory
#setwd(dir) 
# Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("Data", showWarnings=F)
dat_dir <- "./Data"


# OFRA data download to "./Data" (~200 MB!!!)
download("https://www.dropbox.com/s/ojnndjn88urhq6w/PCA_22.zip?dl=0", "./Data/PCA_22.zip", mode="wb")

#import tiff files of PC scores 
unzip("./Data/PCA_22.zip",exdir="./Data", junkpaths=TRUE, overwrite=TRUE)
glist2 <- list.files(path="./Data", pattern="tif", full.names=TRUE)
grids <- stack(glist2)

#Download maize experiments
download("https://www.dropbox.com/s/110f2lnexpwp7vk/OFRA_maize_2014.zip?dl=0", "./Data/OFRA_maize_2014.zip", mode="wb")
#unzip maize experiments
unzip("./Data/OFRA_maize_2014.zip", exdir="./Data", junkpaths=TRUE, overwrite=TRUE)
maize_tr_laea=readShapeSpatial(paste(dat_dir, "OFRA_maize_2014.shp", sep=""))
proj4string(maize_tr_laea)=CRS("+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")
maize_tr=extract(grids2, maize_tr_laea)
maize_tr2=as.data.frame(maize_tr_laea)
maize=maize_tr2[,1:2]
maize_locs=cbind(maize,maize_tr)#file with locations and variables
#divide in training and test data
set.seed(12345)
samp=sample(nrow(maize_locs), round(0.75*nrow(maize_locs)))
traindata=maize_locs[samp,]
train=traindata[,1:2]
traindata=traindata[,3:5]
testdata=maize_locs[-samp,]
test=testdata[,1:2]
testdata=testdata[,3:5]

#test on 200 samples in 50 km radius around 100 Maize experiments
#download shapefile maize_50k_random.zip
download("https://www.dropbox.com/s/tlrrof3faqjbtqc/maize_50k_random.zip?dl=0", "./Data/MW_random_laea.zip", mode="wb")
maize_random=readShapeSpatial(paste(dat_dir, "Maize_50k_random.shp", sep="")
gback=extract(grids2,maize_random)
maize_random=as.data.frame(maize_random)
maize_b=maize_random[,1:2]
#listing presence (1) and background (0)
pb <- c(rep(1, nrow(maize)), rep(0, nrow(maize_b)))
presback <- data.frame(cbind(pb, rbind(maize_tr, gback)))

#statistical tests
#mahalanobis on traindata
#use k-fold partitioning on train data OFRA
pres=presback[presback[,1]==1,2:4]
back=presback[presback[,1]==0, 2:4]
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

#avoiding or removing spatial sorting bias on OFRA maize
nr <- nrow(maize)
s <- sample(nr, 0.25 * nr)
pres_train <- maize[-s, ]
pres_test <- maize[s, ]
nr <- nrow(maize_b)
s <- sample(nr, 0.25 * nr)
back_train <- maize_b[-s, ]
back_test <- maize_b[s, ]
sb <- ssb(pres_test, back_test, pres_train)
sb[,1] / sb[,2] #if p is close to 1, sorting bias is very low

#mahalanobis on 75% train data
nr <- nrow(maize)
s <- sample(nr, 0.75 * nr)
pres_train <- maize[-s, ]
pres_test <- maize[s, ]
nr <- nrow(maize_b)
s <- sample(nr, 0.75 * nr)
back_train <- maize_b[-s, ]
back_test <- maize_b[s, ]
mh=mahal(grids2, pres_train)
e=evaluate(pres_test, back_test, mh, grids2) #evaluate test presence data with background test data
plot(e,'ROC')
tr=threshold(e, 'spec_sens')
pm=predict(grids2,mh, progress= '')
pmb=pm
pmb[pmb<=tr]=tr
plot(pmb)

#mahalanobis all data
mh=mahal(grids2, pres)
e=evaluate(pres_test, back_test, mh, grids2)
pm2=predict(grids2,mh, progress= '')
tr=threshold(e, 'spec_sens')
pm2[pm2<0]=0
plot(pm2)


