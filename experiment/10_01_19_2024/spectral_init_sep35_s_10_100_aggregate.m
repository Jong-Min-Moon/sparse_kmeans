s_trend = [spectralinitp3000strend1 spectralinitp3000strend2];
s_vec = (1:10)*100;
s_vec = s_vec';
mat_for_tex = [s_vec table2array( mean(s_trend, 1) )'];
csvwrite("spectral_init_p_3000_s_trend_macro.csv", mat_for_tex)