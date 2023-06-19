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

###########################################################################################################
######################################### Brute Force optimization ########################################
###########################################################################################################
#result_presWAT = run_BruteForce_opt(measure_type=1)
#result_co2sl   = run_BruteForce_opt(measure_type=2)
#result_temp    = run_BruteForce_opt(measure_type=3)
#reuslt_presWAT_co2sl = run_BruteForce_opt(measure_type=4)

###########################################################################################################
########################################### Layer optimization ###########################################
###########################################################################################################
run_layer_opt(measure_type=1, layer=1)
run_layer_opt(measure_type=1, layer=2)
run_layer_opt(measure_type=1, layer=3)

run_layer_opt(measure_type=2, layer=1)
run_layer_opt(measure_type=2, layer=2)
run_layer_opt(measure_type=2, layer=3)

run_layer_opt(measure_type=3, layer=1)
run_layer_opt(measure_type=3, layer=2)
run_layer_opt(measure_type=3, layer=3)

run_layer_opt(measure_type=4, layer=1)
run_layer_opt(measure_type=4, layer=2)
run_layer_opt(measure_type=4, layer=3)

###########################################################################################################
########################################### Column optimization ###########################################
###########################################################################################################
run_column_opt(measure_type=1)
run_column_opt(measure_type=2)
run_column_opt(measure_type=3)
run_column_opt(measure_type=4)    