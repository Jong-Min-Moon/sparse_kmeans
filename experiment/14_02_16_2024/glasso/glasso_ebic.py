import numpy as np
from inverse_covariance import QuicGraphicalLassoEBIC
import scipy.io as sio
model = QuicGraphicalLassoEBIC(lam=1.0, init_method="cov", gamma=0.01)

data = np.load("data_scaled.pkl", allow_pickle=True)

model.fit(data)
savedict = {
    'Omega_est_now' : model.precision_
}

sio.savemat("cov.mat", savedict)
