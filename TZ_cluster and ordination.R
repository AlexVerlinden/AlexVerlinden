# ordination and clustering of TZ crop data
require (downloader)
require (vegan)
require (MASS)
dir.create("TZ_crops", showWarnings=F)
dat_dir <- "./TZ_crops"
# presence/absence locations of animals, buildings, crops as proxy for farming systems
# download data from 2015 crop scout ODK forms
download.file("https://www.dropbox.com/s/7ul7rkooy0xvvgz/Crops_NTZ_short2_zeros.csv?dl=0", "./TZ_crops/Crops_NTZ_short2_zeros.csv", mode="wb")
ord1=read.table(paste(dat_dir, "/Crops_NTZ_short2_zeros.csv", sep=""), header=T, sep=",")
#deleting rare occurences wheat barley rice etc.#
ord2=subset(ord1, select = -c(wheat, sorghum, cereal_pres, crop_pres, livestock_pres, rootcrops_pres, legumes_pres, cereal.other, other_crops, legume.other, root.other, barley, rice, livestock.other, chickpea, millet, camel))


#ordination using vegan
ord=na.omit(ord2)
#removing rows with only zeros
ord= ord[!!rowSums(abs(ord[-c(1:2)])),]


#ordination with CA, note do not remove duplicates
ordec=cca(ord)
pl <- plot(ordec, dis="sp")

#cluster
#removing duplicates only needed for clustering
orduniq <- unique(ord)
orduniq=na.omit(orduniq)
#distance matrix
ord_diss=vegdist(orduniq, method="bray")

clua <- hclust(ord_diss, "average")
plot(clua, cex=0.4)
rect.hclust(clua, 6)
grp=cutree(clua, 6)
