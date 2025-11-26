
library(sparcl)
library(RSQLite)

source("competitor.R")
source("data_generation.R")
source("get_proportional_subsample.R")


tryCatch({
  # Adjust sep="" if your file uses a different delimiter (e.g., sep=",")
  lx.original <- read.table("leuk_x.txt", header = FALSE)[, 1]
  message("Successfully loaded leuk_x.txt.")
}, error = function(e) {
  # Provide an error message if the file cannot be found or read
  stop(paste("Error reading leuk_x.txt:", e$message, "Please ensure the file exists and contains valid data."))
})

# 2. Read the data for y (ly.original)
tryCatch({
  ly.original <- read.table("leuk_y.txt", header = FALSE)[, 1]
  message("Successfully loaded leuk_y.txt.")
}, error = function(e) {
  stop(paste("Error reading leuk_y.txt:", e$message, "Please ensure the file exists and contains valid data."))
})
x_full <- 2^lx.original 
y_full <- ly.original
y_full= c(y_full$label)
print(dim(x_full))
print(dim(y_full))
# Initialize sums for accuracy and timing

accuracy_witten = rep(0,100)
for (rep in 1:100){
  data = get_proportional_subsample(x_full, y_full, 1000, i)
  start_time_witten = Sys.time()
  sol_witten = sparse_kmeans_ell1_ell2(data$X)
  end_time_witten = Sys.time()
  time_witten = as.numeric(end_time_witten - start_time_witten, units="secs")
  accuracy_witten[rep] <- evaluate_clustering(sol_witten, data$y)
}
    



