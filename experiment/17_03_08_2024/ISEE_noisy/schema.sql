DROP TABLE IF EXISTS ISEE_noisy;
CREATE TABLE ISEE_noisy (
  iter_num TEXT PRIMARY KEY,
  inde_rep INTEGER NOT NULL,
  iter INTEGER NOT NULL,
  acc REAL NOT NULL,
  true_discov INTEGER NOT NULL, 
  false_discov INTEGER NOT NULL,
  diff_x_tilde REAL NOT NULL,
  diff_omega_diag REAL NOT NULL,
  omega_est_time REAL NOT NULL,
  sdp_solve_time REAL NOT NULL,
);


