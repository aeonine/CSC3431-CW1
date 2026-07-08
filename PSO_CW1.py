# Import modules
import itertools
from multiprocessing import Pool
import pathlib
import numpy as np
import pandas as pd
import time 

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

# Definition of the problem bounds, do not touch this block of code
ndim = 10
max_bound = 5.12 * np.ones(ndim)
min_bound = - max_bound
bounds = (min_bound, max_bound)

logFile = str("pso_third_data")

def initParams():
	params = {
		"c1": 0.3,
		"c2": 0.5,
		"w" : 1,
		"p" : 5,
		"k" : 2,
		"particles":100,
		"iterations":100,
	}
	return params

def runPSO(params):

	# Call instance of PSO
	if(params["topology"] == "localBest"):
		optimiser = ps.single.LocalBestPSO(n_particles=params["particles"], dimensions=ndim, bounds=bounds, options=params)
	if(params["topology"] == "globalBest"):
		optimiser = ps.single.GlobalBestPSO(n_particles=params["particles"], dimensions=ndim, bounds=bounds, options=params)
	# Perform optimization
	cost, pos = optimiser.optimize(fx.rastrigin, iters=params["iterations"],verbose=False)

	#print("Best solution: {}, fitness value: {}".format(pos,cost))
	return cost,optimiser.cost_history
	#print(optimiser.cost_history)

def runExperiment(params,logFile,label):
	# time the script
	start = time.time()
	
	nreps = 30
	best_reps = []
	best_its = {}
	for rep in range(nreps):
		bestFit,best_it = runPSO(params)
		#print("Best fitness of repetiton {} : {}".format(rep,bestFit))
		best_reps.append(bestFit)
	
		for it in range(len(best_it)):
			if not it in best_its:
				best_its[it] = []
			best_its[it].append(best_it[it])
	
	trace = []
	for it in sorted(best_its.keys()):
		trace.append([it,np.mean(best_its[it])] + label.split(","))

	column_keys = ["It","AveBestFitness"]
	banned_keys = ['mutation', 'selection', 'crossover']

	for pkey in params.keys():
		if(pkey not in banned_keys):
			column_keys.append(pkey)

	data_path = pathlib.Path("./data/"+logFile+"/")
	data_path.mkdir(exist_ok=True, parents=True)
		
	tdf = pd.DataFrame(trace, columns=column_keys)
	tdf.to_csv("./data/" + logFile + "/"+ label + ".csv",index=False,mode="x") 
	
	print("Average of best fitness for experiment {} : {}".format(label,np.mean(best_reps)))
	
	end = time.time()
	print("Average run time for experiment {} : {}".format(label,(end-start)/nreps))

params = initParams()

def tidy_possibility(pset):

	if(pset["iterations"] < 100):
		pset["particles"] = 200 - pset["particles"]
			
	if(pset["particles"] < 100):
		pset["iterations"] = 200 - pset["particles"]
		
	if(pset["iterations"] == 100):
		pset["particles"] == 100

	if(pset["particles"] == 100):
		pset["iterations"] == 100

	return pset
	
	
def create_column_label(pms):
	temp_string = str()
	for key in pms.keys():
		temp_string.__add__("," + str(key))
		
	return temp_string

def create_experiment_label(pms):
	temp_string = ""
	first = True
	for key, value in pms.items():
		if(key == "selection" or key == "mutation" or key == "crossover"):
			next
		else:
			if not first:
				temp_string += ("," + str(value))
			if first:
				temp_string += (str(value))
				first = False
	return temp_string

def generate_parameter_tuple(map, logFile):
	temp_list = list()
	for pm in map:
		temp_list.append((pm, logFile, create_experiment_label(pm)))

	return tuple(temp_list)

# CODE BEYOND THIS POINT IS FOR INITIALIZATION


paramdict_initial = {
		"c1":[0.25,0.3, 0.5, 0.75, 1],
		"c2":[0.25,0.3, 0.5, 0.75, 1],
		"w":[0.25, 0.5, 1],
		"p" : [2.5,5,10],
		"k" : [2,5,10],
		"particles": [100],
		"iterations": [100],
		"topology" : ["localBest"]

}

paramdict_second = {
		"c1":[0.75, 1],
		"c2":[1],
		"w":[0.5, 0.75],
		"p" : [4,5,6],
		"k" : [4,5,6],
		"particles": [100],
		"iterations": [100],
		"topology" : ["localBest"]

}

paramdict_third = {
		"c1":[0.75, 1],
		"c2":[1],
		"w":[0.5, 0.75],
		"p" : [4,5,6],
		"k" : [4,5,6],
		"particles": [50],
		"iterations": [150],
		"topology" : ["globalBest"]
}

paramdict_fourth = {
		"c1":[1],
		"c2":[1, 1.25],
		"w":[0.05, 0.065, 0.75],
		"p" : [6],
		"k" : [4,5,6],
		"particles": [50],
		"iterations": [150],
		"topology" : ["globalBest"]
}

def run_initial_experiment():
	
	folderPath = str("pso_initial_data")

	paramkey, paramvalue = zip(*paramdict_initial.items())
	paramset_initial = []
	paramset_initial = tuple(dict(zip(paramkey, paramver)) for paramver in itertools.product(*paramvalue) if dict(zip(paramkey, paramver)) not in paramset_initial)
	parammap_initial = []

	with Pool(processes=64) as pool:
		parammap_initial.append(pool.map(tidy_possibility, paramset_initial))

	parammap_initial = tuple(parammap_initial[0])

	init_tuple = generate_parameter_tuple(parammap_initial, logFile=folderPath)

	with Pool(processes=64) as pool:
		pool.starmap(runExperiment, init_tuple)

def run_second_experiment():
	folderPath = str("pso_second_data")

	paramkey, paramvalue = zip(*paramdict_second.items())
	paramset_second = []

	paramset_second = tuple(dict(zip(paramkey, paramver)) for paramver in itertools.product(*paramvalue) if dict(zip(paramkey, paramver)) not in paramset_second)
	parammap_second = []

	with Pool(processes=64) as pool:
		parammap_second.append(pool.map(tidy_possibility, paramset_second))

	parammap_second = tuple(parammap_second[0])

	second_tuple = generate_parameter_tuple(parammap_second, logFile=folderPath)

	with Pool(processes=64) as pool:
		pool.starmap(runExperiment, second_tuple)

def run_third_experiment():
	folderPath = str("pso_third_data")

	paramkey, paramvalue = zip(*paramdict_third.items())
	paramset_third = []
	paramset_third = tuple(dict(zip(paramkey, paramver)) for paramver in itertools.product(*paramvalue) if dict(zip(paramkey, paramver)) not in paramset_third)
	parammap_third = []

	with Pool(processes=64) as pool:
		parammap_third.append(pool.map(tidy_possibility, paramset_third))

	parammap_third = tuple(parammap_third[0])

	third_tuple = generate_parameter_tuple(parammap_third, logFile=folderPath)

	with Pool(processes=64) as pool:
		pool.starmap(runExperiment, third_tuple)

def run_fourth_experiment():
	folderPath = str("pso_fourth_data")

	paramkey, paramvalue = zip(*paramdict_fourth.items())
	paramset_fourth = []
	paramset_fourth = tuple(dict(zip(paramkey, paramver)) for paramver in itertools.product(*paramvalue) if dict(zip(paramkey, paramver)) not in paramset_fourth)
	parammap_fourth = []

	with Pool(processes=64) as pool:
		parammap_fourth.append(pool.map(tidy_possibility, paramset_fourth))

	parammap_fourth = tuple(parammap_fourth[0])

	fourth_tuple = generate_parameter_tuple(parammap_fourth, logFile=folderPath)

	with Pool(processes=64) as pool:
		pool.starmap(runExperiment, fourth_tuple)

def run_all_experiments():

	run_initial_experiment()
	run_second_experiment()

run_third_experiment()