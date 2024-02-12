import numpy as np
from inverse_covariance import QuicGraphicalLassoEBIC
import scipy.io as sio
model = QuicGraphicalLassoEBIC(lam=1.0, init_method="cov", gamma=0.01)

