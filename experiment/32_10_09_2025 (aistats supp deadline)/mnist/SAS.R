
library(sparcl)
library(RSQLite)

source("competitor.R")
source("data_generation.R")
source("get_proportional_subsample.R")


# Load full dataset. please convert the attatched .mat file into csv. We cannot upload the converted csv on github due to file size limits.
x_full <- read.csv(".csv") # get data from attatched autoencoder.ipynb
y_full <- read.csv(".csv")
y_full= c(y_full$label)
print(dim(x_full))
print(dim(y_full))
# Initialize sums for accuracy and timing

accuracy_arias = rep(0,100)
for (rep in 1:100){
  data = get_proportional_subsample(x_full, y_full, 1000, i)
  start_time_witten = Sys.time()

  
  sol_arias = sparse_kmeans_hillclimb(data$X)
  end_time_arias = Sys.time()
  time_arias  =  as.numeric(end_time_arias - start_time_arias, units="secs")
  accuracy_arias[rep] <- evaluate_clustering(sol_arias, data$y)
}
    



