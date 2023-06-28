from Uncertainty import *
from uncertaintyMetric import *
from optim_utils import *

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

##### Brute Force optimization #####
# result_presWAT = run_BruteForce_opt(measure_type=1)
# result_co2sl   = run_BruteForce_opt(measure_type=2)
# result_temp    = run_BruteForce_opt(measure_type=3)
# reuslt_presWAT_co2sl = run_BruteForce_opt(measure_type=4)

##### Layer optimization #####
# run_layer_opt(measure_type=1, layer=1); run_layer_opt(measure_type=1, layer=2); run_layer_opt(measure_type=1, layer=3)
# run_layer_opt(measure_type=2, layer=1); run_layer_opt(measure_type=2, layer=2); run_layer_opt(measure_type=2, layer=3)
# run_layer_opt(measure_type=3, layer=1); run_layer_opt(measure_type=3, layer=2); run_layer_opt(measure_type=3, layer=3)
# run_layer_opt(measure_type=4, layer=1); run_layer_opt(measure_type=4, layer=2); run_layer_opt(measure_type=4, layer=3)

##### Column optimization #####
# run_column_opt(measure_type=1); run_column_opt(measure_type=2); run_column_opt(measure_type=3); run_column_opt(measure_type=4)    


#########################################################################################################
#########################################################################################################
### LAYER OPTIMIZATION
'''
layer1, layer2, layer3 = np.arange(2,18), np.arange(18,34), np.arange(34,50)
layer_list = [layer1, layer2, layer3]
result = pd.DataFrame(index=['layer 1', 'layer 2', 'layer 3'], columns=['measure 1', 'measure 2', 'measure 3', 'measure 4'])
for m in range(4):
    for l in range(len(layer_list)):
        result.iloc[l,m] = Proxy(ncol_data=list(layer_list[l]), measure_type=m+1, rom_data=LinearRegression(), rom_obj=LinearRegression(), error_option=2, verbose=True).value
        print('layer {} DONE'.format(l))
    print('measure {} DONE'.format(m))
result.to_csv('optimization_by_layer.csv')
'''

### COLUMN/WELL OPTIMIZATION
well_names = ['well {}'.format(p+1) for p in range(16)]
templist, well = np.arange(2,50,step=16), {}
for i in range(16):
    well[i] = templist+i
wells = np.array(list(well.values()))
results = pd.DataFrame(index=well_names, columns=['measure 1', 'measure 2', 'measure 3', 'measure 4'])
for m in range(4):
    for i in range(16):
        results.iloc[i,m] = Proxy(ncol_data=list(wells[i]), measure_type=m+1, nMCSamples=10000, rom_data=Earth(), rom_obj=Earth(), verbose=True).value
        print('column/well {} DONE'.format(i))
    print('measure {} DONE'.format(m))
results.to_csv('optimization_by_column.csv')

#########################################################################################################
#########################################################################################################