import numpy as np
from inverse_covariance import QuicGraphicalLassoEBIC
import scipy.io as sio
model = QuicGraphicalLassoEBIC(lam=1.0, init_method="cov", gamma=0.01)

data = np.load('/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/rho20_Delta5_p100.pkl', allow_pickle=True)
model.fit(data)
savedict = {'Omega_est_now' : model.precision_}
sio.savemat('/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/rho20_Delta5_p100.mat', savedict)
