b=[3,4,5]
dir = '/home1/jongminm/sparse_kmeans/experiment/14_02_16_2024/ISEE'
dir
addpath(genpath(dir))
a = [1, 2, 3]
csvwrite( strcat(dir, '/result.csv'), a)
