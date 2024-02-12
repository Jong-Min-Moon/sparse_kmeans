import numpy as np
from inverse_covariance import QuicGraphicalLassoEBIC

import scipy.io as sio
data = np.load("/media/jongmin/data/GitHub/sparse_kmeans/experiment/14_02_16_2024/glasso/data_scaled.pkl", allow_pickle=True)
model = QuicGraphicalLassoEBIC(lam=1.0, init_method="cov", gamma=0.01)
model.fit(data)
savedict = {
    'Omega_est_now' : model.precision_
}
sio.savemat("/media/jongmin/data/GitHub/sparse_kmeans/experiment/14_02_16_2024/glasso/cov.mat", savedict)
