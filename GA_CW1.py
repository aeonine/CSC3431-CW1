import csv
import random
import time
from deap import creator, base, tools, algorithms, benchmarks
import numpy as np
import pandas as pd
import itertools
from multiprocessing import Pool, Process
import pathlib

# A positive weight indicates a maximisation function (higher is better), a negative weight a minimisation function (lower is better)
# The settings in the next block of lines define the problem to solve, so you cannot modify them
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_double", random.uniform, -5.12, 5.12)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_double, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", benchmarks.rastrigin)

def initParams():
	params = {
		"popSize":100,
		"iterations":100,
		"pcross":0.6,
		"pmut_ind":0.3,
		"pmut_gene":0.075,
		"tournsize":10,
		"mu":0,
		"sigma":0.5,
		"cx_uniform":5,
		"selection":tools.selTournament,
		"crossover":tools.cxUniformPartialyMatched,
		"mutation":tools.mutUniformInt
	}
	return params

def runGA(params):

	if(params["crossover"] == "cxUniformPartiallyMatched"):
		toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=params["pmatch_rate"])
	
	if(params["crossover"] == tools.cxBlend):
		toolbox.register("mate", tools.cxBlend, alpha=params["alpha"])
	
	if(params["crossover"] == tools.cxESBlend):
		toolbox.register("mate", tools.cxBlend, alpha=params["alpha"])

	if(params["crossover"] == "default"):
		toolbox.register("mate", tools.cxOnePoint)

	if(params["crossover"] == tools.cxOrdered):
		toolbox.register("mate", tools.cxOrdered)

	if(params["crossover"] == tools.cxSimulatedBinaryBounded):
		toolbox.register("mate", tools.cxSimulatedBinaryBounded, up=params["up"], low=params["low"], eta=params["tournsize"])
	
	if(params["mutation"] == "mutUniformInt"):
		toolbox.register("mutate", tools.mutUniformInt, up=params["up"], low=params["low"], indpb=params["indpb"])

	if(params["mutation"] == "default"):
		toolbox.register("mutate", tools.mutGaussian, mu=params["mu"], sigma=params["sigma"], indpb=params["pmut_gene"])
	
	if(params["mutation"] == tools.mutPolynomialBounded):
		toolbox.register("mutate", tools.mutPolynomialBounded, up=params["up"], low=params["low"], eta=params["tournsize"], indpb=params['pmut_gene'])

	if(params["selection"] == tools.selBest):
		toolbox.register("select", tools.selBest, k=params["tournsize"])

	if(params["selection"] == tools.selWorst):
		toolbox.register("select", tools.selWorst, k=params["tournsize"])

	if(params["selection"] == tools.selRoulette):
		toolbox.register("select", tools.selRoulette, k=params["tournsize"])
	else:
		toolbox.register("select", tools.selTournament, tournsize=params["tournsize"])

	# These define the operators of the GA, and can be selected/tuned for the problem at hand
	# This is the population size, the number of alternative solutions being evolved
	popSize=params["popSize"]
	population = toolbox.population(n=popSize)
	hof = tools.HallOfFame(1)

	# Evaluate the initial population
	fits = toolbox.map(toolbox.evaluate, population)
	for fit, ind in zip(fits, population):
		ind.fitness.values = fit

	hof.update(population)

	best_it = []
	# Number of iterations of the evolutionary cycle
	NGEN=params["iterations"]

	for gen in range(NGEN):
		population = toolbox.select(population, k=len(population))
		offspring = algorithms.varAnd(population, toolbox, cxpb=params["cxpb"], mutpb=params["pmut_ind"])

		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		population[:] = offspring

		hof.update(population)
		top1 = tools.selBest(population, k=1)
		best_it.append(top1[0].fitness.values[0])
	
	best = hof[0]
	#print("Best individual {} -> fitness: {}".format(best,best.fitness.values[0]))
	return best.fitness.values[0],best_it


def runExperiment(params,logFile,label):
	# time the script
	start = time.time()
	
	nreps = 30
	best_reps = []
	best_its = {}
	for rep in range(nreps):
		bestFit,best_it = runGA(params)
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
	#print("Average run time for experiment {} : {}".format(label,(end-start)/nreps))

def tidy_possibility(pset):

	if(pset["iterations"] < 100):
		pset["popSize"] = 200 - pset["iterations"]
			
	if(pset["popSize"] < 100):
		pset["iterations"] = 200 - pset["popSize"]
		
	if(pset["iterations"] == 100):
		pset["popSize"] == 100
	
	if("low" and "up" in pset.keys()):
		if(pset["low"] == pset["up"]):
			pset["low"] = round(pset["up"]/2)
		
		if(pset["low"] == 1 and pset["up"] == 1):
			pset["up"] = 2

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
		"popSize":[100],
		"iterations":[100],
		"pcross":[0.5, 0.6],
		"pmut_ind":[0.1, 0.25, 0.3],
		"pmut_gene":[0.05, 0.075, 0.1],
		"tournsize":[2,3,5,10,20],
		"cxpb":[0,1],
		"indpb" : [5, 10, 15],
		"sigma":[0.1, 0.5, 1],
		"selection":[tools.selTournament],
		"crossover":["default"],
		"mutation":["default"]
}


paramdict_second = {
		"popSize":[50],
		"iterations":[150],
		"pcross":[0.05, 0.075],
		"pmut_ind":[0.5],
		"pmut_gene":[0.05, 0.1],
		"tournsize":[2,3,5,10],
		"cxpb":[0,1],
		"indpb" : [25, 100],
		"sigma":[0.1, 0.5, 1],
		"selection":[tools.selTournament],
		"crossover":["default"],
		"mutation":["default"]
}

paramdict_third = {
		"popSize":[50],
		"iterations":[150],
		"pcross":[0.06, 0.075, 0.08],
		"pmut_ind":[0.25, 0.35, 0.5],
		"pmut_gene":[0.05, 0.65, 0.7],
		"tournsize":[5,10],
		"cxpb":[0,1],
		"indpb" : [25, 100],
		"sigma":[0.05, 0.1, 0.15],
		"selection":[tools.selTournament],
		"crossover":["default"],
		"mutation":["default"]
}

paramdict_fourth = {
		"popSize":[50],
		"iterations":[150],
		"pcross":[0.04, 0.045, 0.05],
		"pmut_ind":[0.35, 0.4, 0.5],
		"pmut_gene":[0.45, 0.5, 0.65],
		"tournsize":[5,10],
		"cxpb":[0,1],
		"indpb" : [75, 100],
		"sigma":[0.15, 0.20, 0.25],
		"selection":[tools.selTournament],
		"crossover":["default"],
		"mutation":["default"]
}

paramdict_fifth = {
		"popSize":[50],
		"iterations":[150],
		"pcross":[0.05, 0.075],
		"pmut_ind":[0.25, 0.5],
		"pmut_gene":[0.05, 0.1],
		"tournsize":[10,15,20],
		"cxpb":[0.85,1],
		"indpb" : [25, 100],
		"sigma":[0.1, 0.5, 1],
		"low":[-5.12],
		"up":[5.12],
		"selection":[tools.selTournament],
		"crossover":[tools.cxSimulatedBinaryBounded],
		"mutation":["default"]
}

paramdict_sixth = {
		"popSize":[50],
		"iterations":[150],
		"pcross":[0.05, 0.075],
		"pmut_ind":[0.1, 0.25, 0.5],
		"pmut_gene":[0.05, 0.1],
		"tournsize":[10,15,20],
		"cxpb":[0.85,1],
		"indpb" : [25, 100],
		"sigma":[0.1, 0.5, 1],
		"low":[-5.12],
		"up":[5.12],
		"selection":[tools.selTournament],
		"crossover":[tools.cxSimulatedBinaryBounded],
		"mutation":[tools.mutPolynomialBounded]
}

paramdict_final = {
		"popSize":[50],
		"iterations":[150],
		"pcross":[0.075],
		"pmut_ind":[0.5],
		"pmut_gene":[0.1],
		"tournsize":[10],
		"cxpb":[1],
		"indpb" : [25],
		"sigma":[0.1],
		"low":[-5.12],
		"up":[5.12],
		"selection":[tools.selTournament],
		"crossover":[tools.cxSimulatedBinaryBounded],
		"mutation":[tools.mutPolynomialBounded]
}

def run_initial_experiment():
	folderPath = str("ga_initial_data")

	root_path = pathlib.Path("./data/")
	root_path.mkdir(exist_ok=True)

	paramkey, paramvalue = zip(*paramdict_initial.items())
	paramset_initial = []

	paramset_initial = tuple(dict(zip(paramkey, paramver)) for paramver in itertools.product(*paramvalue) if dict(zip(paramkey, paramver)) not in paramset_initial)
	parammap_initial = []

	with Pool(processes=64) as pool:
		parammap_initial.append(pool.map(tidy_possibility, paramset_initial))

	parammap_initial = tuple(parammap_initial[0])

	init_tuple = generate_parameter_tuple(parammap_initial, logFile=folderPath)


	with Pool(processes=32) as pool:
		pool.starmap(runExperiment, init_tuple)

	print("Initial Simulation Complete")


def run_second_experiment():

	folderPath = str("ga_second_data")


	root_path = pathlib.Path("./data/")
	root_path.mkdir(exist_ok=True)

	paramkey, paramvalue = zip(*paramdict_second.items())
	paramset_second = []

	paramset_second = tuple(dict(zip(paramkey, paramver)) for paramver in itertools.product(*paramvalue) if dict(zip(paramkey, paramver)) not in paramset_second)
	parammap_second = []

	with Pool(processes=64) as pool:
		parammap_second.append(pool.map(tidy_possibility, paramset_second))

	parammap_second = tuple(parammap_second[0])

	second_tuple = generate_parameter_tuple(parammap_second, logFile=folderPath)

	with Pool(processes=32) as pool:
		pool.starmap(runExperiment, second_tuple)

	print("Second Simulation Complete")

def run_third_experiment():
	folderPath = str("ga_third_data")

	root_path = pathlib.Path("./data/")
	root_path.mkdir(exist_ok=True)
	
	paramkey, paramvalue = zip(*paramdict_third.items())
	paramset_third = []

	paramset_third = tuple(dict(zip(paramkey, paramver)) for paramver in itertools.product(*paramvalue) if dict(zip(paramkey, paramver)) not in paramset_third)
	parammap_third = []

	with Pool(processes=64) as pool:
		parammap_third.append(pool.map(tidy_possibility, paramset_third))

	parammap_third = tuple(parammap_third[0])

	third_tuple = generate_parameter_tuple(parammap_third, logFile=folderPath)

	with Pool(processes=32) as pool:
		pool.starmap(runExperiment, third_tuple)

	print("Third Simulation Complete")

def run_fourth_experiment():
	folderPath = str("ga_fourth_data")

	root_path = pathlib.Path("./data/")
	root_path.mkdir(exist_ok=True)

	paramkey, paramvalue = zip(*paramdict_fourth.items())
	paramset_fourth = []

	paramset_fourth = tuple(dict(zip(paramkey, paramver)) for paramver in itertools.product(*paramvalue) if dict(zip(paramkey, paramver)) not in paramset_fourth)
	parammap_fourth = []

	with Pool(processes=64) as pool:
		parammap_fourth.append(pool.map(tidy_possibility, paramset_fourth))

	parammap_fourth = tuple(parammap_fourth[0])

	fourth_tuple = generate_parameter_tuple(parammap_fourth, logFile=folderPath)

	with Pool(processes=32) as pool:
		pool.starmap(runExperiment, fourth_tuple)

	print("Fourth Simulation Complete")

def run_fifth_experiment():
	folderPath = str("ga_fifth_data")

	root_path = pathlib.Path("./data/")
	root_path.mkdir(exist_ok=True)

	paramkey, paramvalue = zip(*paramdict_fifth.items())
	paramset_fifth = []

	paramset_fifth = tuple(dict(zip(paramkey, paramver)) for paramver in itertools.product(*paramvalue) if dict(zip(paramkey, paramver)) not in paramset_fifth)
	parammap_fifth = []

	with Pool(processes=64) as pool:
		parammap_fifth.append(pool.map(tidy_possibility, paramset_fifth))

	parammap_fifth = tuple(parammap_fifth[0])

	fifth_tuple = generate_parameter_tuple(parammap_fifth, logFile=folderPath)

	with Pool(processes=32) as pool:
		pool.starmap(runExperiment, fifth_tuple)

	print("Fourth Simulation Complete")

def run_sixth_experiment():
	folderPath = str("ga_sixth_data")

	root_path = pathlib.Path("./data/")
	root_path.mkdir(exist_ok=True)

	paramkey, paramvalue = zip(*paramdict_sixth.items())
	paramset_sixth = []

	paramset_sixth = tuple(dict(zip(paramkey, paramver)) for paramver in itertools.product(*paramvalue) if dict(zip(paramkey, paramver)) not in paramset_sixth)
	parammap_sixth = []

	with Pool(processes=64) as pool:
		parammap_sixth.append(pool.map(tidy_possibility, paramset_sixth))

	parammap_sixth = tuple(parammap_sixth[0])

	sixth_tuple = generate_parameter_tuple(parammap_sixth, logFile=folderPath)

	# run the process batch in a pool
	with Pool(processes=32) as pool:
		pool.starmap(runExperiment, sixth_tuple)

	print("Sixth Simulation Complete")

def run_final_experiment():
	folderPath = str("ga_final_data")

	root_path = pathlib.Path("./data/")
	root_path.mkdir(exist_ok=True)

	paramkey, paramvalue = zip(*paramdict_final.items())

	paramset_final = []
	paramset_final = tuple(dict(zip(paramkey, paramver)) for paramver in itertools.product(*paramvalue) if dict(zip(paramkey, paramver)) not in paramset_final)
	
	parammap_final = []

	with Pool(processes=64) as pool:
		parammap_final.append(pool.map(tidy_possibility, paramset_final))

	parammap_final = tuple(parammap_final[0])

	final_tuple = generate_parameter_tuple(parammap_final, logFile=folderPath)

	# run the process batch in a pool
	with Pool(processes=32) as pool:
		pool.starmap(runExperiment, final_tuple)

	print("Final Simulation Complete")


def run_all_simulations():
	run_initial_experiment()
	run_second_experiment()
	run_third_experiment()
	run_fourth_experiment()
	run_fifth_experiment()
	run_sixth_experiment()
	run_final_experiment()

run_all_simulations()