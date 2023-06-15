from Uncertainty import *
from uncertaintyMetric import *
from utils import *

K.clear_session()
check_tensorflow_gpu()

# select       = 7
# plot_bounds  = [-3, 3]
# local_method = 'trust-krylov'
# x0           = [1.5, 1.5]
# glob_bounds  = [-2, 2]
# opt = FullOpt(select)
# local_res,  local_res_df  = opt.local_opt(x0=x0, method=local_method)
# global_res, global_res_df = opt.global_opt(varbounds=glob_bounds)
# opt.make_plot(global_res_df, local_res_df, mbounds=plot_bounds, angle=[45,225],
#               showcontours=True, showtrajectory=True)

result_presWAT = {}
opt = FullOpt(select=0, dims=1)
opt.measure_type     = 1
opt.nDataRealization = 200
opt.nMCSamples       = 100000
opt.ROM_data, opt.ROM_obj = Earth(), Earth()
for i in range(2, 50):
    result_presWAT[i] = opt.fun(i)
result_presWAT_arr = np.array(list(result_presWAT.values()))
np.save('result_presWAT.npy', result_presWAT_arr)
min, minloc = result_presWAT_arr.min(), result_presWAT_arr.argmin()
print('Minima: {:.3f} x 1e6 | Column #: {}'.format(min, minloc))

result_co2sl = {}
opt = FullOpt(select=0, dims=1)
opt.measure_type     = 2
opt.nDataRealization = 200
opt.nMCSamples       = 100000
opt.ROM_data, opt.ROM_obj = Earth(), Earth()
for i in range(2, 50):
    result_co2sl[i] = opt.fun(i)
result_co2sl_arr = np.array(list(result_co2sl.values()))
np.save('result_co2sl.npy', result_co2sl_arr)
min, minloc = result_co2sl_arr.min(), result_co2sl_arr.argmin()
print('Minima: {:.3f} x 1e6 | Column #: {}'.format(min, minloc))

result_temp = {}
opt = FullOpt(select=0, dims=1)
opt.measure_type     = 3
opt.nDataRealization = 200
opt.nMCSamples       = 100000
opt.ROM_data, opt.ROM_obj = Earth(), Earth()
for i in range(2, 50):
    result_temp[i] = opt.fun(i)
result_temp_arr = np.array(list(result_temp.values()))
np.save('result_temp.npy', result_temp_arr)
min, minloc = result_temp_arr.min(), result_temp_arr.argmin()
print('Minima: {:.3f} x 1e6 | Column #: {}'.format(min, minloc))

result_presWAT_co2sl = {}
opt = FullOpt(select=0, dims=1)
opt.measure_type     = 4
opt.nDataRealization = 200
opt.nMCSamples       = 100000
opt.ROM_data, opt.ROM_obj = Earth(), Earth()
for i in range(2, 50):
    result_presWAT_co2sl[i] = opt.fun(i)
result_presWAT_co2sl_arr = np.array(list(result_presWAT_co2sl.values()))
np.save('result_presWAT_co2sl.npy', result_presWAT_co2sl_arr)
min, minloc = result_presWAT_co2sl_arr.min(), result_presWAT_co2sl_arr.argmin()
print('Minima: {:.3f} x 1e6 | Column #: {}'.format(min, minloc))