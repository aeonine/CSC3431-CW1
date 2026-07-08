import pathlib
import pandas
import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

def generate_graph_for_pso_dataset(dir, select, outputdir, label):

    root_path = pathlib.Path(outputdir)
    root_path.mkdir(exist_ok=True, parents=True)

    df = pandas.DataFrame()
    scatter = plt.figure()
    ax = scatter.add_subplot(111, projection='3d')

    root_path = pathlib.Path(dir)
    root_path.mkdir(exist_ok=True, parents=True)

    data = []
    for file in os.listdir(dir):
        if(file.endswith(".csv")):
            data.append(pandas.read_csv(dir + file))
    
    df = pandas.concat(data, ignore_index=True)

    ax.scatter(df['It'], df[select], df['AveBestFitness'])
    
    best_fitness_index = df['AveBestFitness'].idxmin()
    best_paramset= df.loc[best_fitness_index]

    ax.scatter(
        best_paramset['It'],
        best_paramset[select],
        best_paramset['AveBestFitness'],
        s=100,
        marker="*",
        edgecolor="k"
    )

    title = "Best Parameter Set\n"

    for param in best_paramset.keys():
        title += param + ":" + str(best_paramset[param]) + "\n"

    ax.text(
        0.99, 0.01, 0.99,
        s=title,
        ha='right',
        va='bottom',
        fontsize=5,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(select)
    ax.set_zlabel("Average Best Fitness")
    plt.tight_layout()
    plt.savefig(outputdir+label+"graph-"+select)

def generate_graph_for_ga_dataset(dir, compare, outputdir, label):

    root_path = pathlib.Path(outputdir)
    root_path.mkdir(exist_ok=True)
    
    df = pandas.DataFrame()
    scatter = plt.figure()
    ax = scatter.add_subplot(111, projection='3d')

    data = []
    for file in os.listdir(dir):
        if(file.endswith(".csv")):
            data.append(pandas.read_csv(dir + file))
    
    df = pandas.concat(data, ignore_index=True)

    ax.scatter(df['It'], df[compare], df['AveBestFitness'])
    
    best_fitness_index = df['AveBestFitness'].idxmin()
    best_paramset= df.loc[best_fitness_index]

    ax.scatter(
        best_paramset['It'],
        best_paramset[compare],
        best_paramset['AveBestFitness'],
        s=100,
        marker="*",
        edgecolor="k"
    )


    ax.set_xlabel("Iteration")
    ax.set_ylabel(compare)
    ax.set_zlabel("Average Best Fitness")

    title = "Best Parameter Set\n"

    for param in best_paramset.keys():
        title += param + ":" + str(best_paramset[param]) + "\n"

    ax.text(
        0.90, 0.5, 0.90,
        s=title,
        ha='left',
        va='center',
        fontsize=5,
    )
    plt.savefig(outputdir+label+"graph-"+compare)

def generate_2d_graph(dir, comparex, comparey, outputdir, label):

    root_path = pathlib.Path(outputdir)
    root_path.mkdir(exist_ok=True)

    data = []
    for file in os.listdir(dir):
        if(file.endswith(".csv")):
            data.append(pandas.read_csv(dir + file))

    df = pandas.concat(data, ignore_index=True)

    plt.plot(df[comparex], df[comparey])
    plt.xlabel(comparex)
    plt.ylabel(comparey)
    plt.savefig(outputdir+label+"2dgraph-"+comparey+comparex)

def generate_graph_initial_pso():
    generate_graph_for_pso_dataset("./data/pso_initial_data/","p", "./graphs/pso_initial_data/", "pso_initial")
    generate_graph_for_pso_dataset("./data/pso_initial_data/","k", "./graphs/pso_initial_data/", "pso_initial")
    generate_graph_for_pso_dataset("./data/pso_initial_data/","c1", "./graphs/pso_initial_data/", "pso_initial")
    generate_graph_for_pso_dataset("./data/pso_initial_data/","c2", "./graphs/pso_initial_data/", "pso_initial")
    generate_graph_for_pso_dataset("./data/pso_initial_data/","w", "./graphs/pso_initial_data/", "pso_initial")

def generate_graph_second_pso():
    generate_graph_for_pso_dataset("./data/pso_second_data/","p", "./graphs/pso_second_data/", "pso_second")
    generate_graph_for_pso_dataset("./data/pso_second_data/","k", "./graphs/pso_second_data/", "pso_second")
    generate_graph_for_pso_dataset("./data/pso_second_data/","w", "./graphs/pso_second_data/", "pso_second")
    generate_graph_for_pso_dataset("./data/pso_second_data/","c1", "./graphs/pso_second_data/", "pso_second")
    generate_graph_for_pso_dataset("./data/pso_second_data/","c2", "./graphs/pso_second_data/", "pso_second")

def generate_graph_third_pso():
    generate_graph_for_pso_dataset("./data/pso_third_data/","p", "./graphs/pso_third_data/", "pso_third")
    generate_graph_for_pso_dataset("./data/pso_third_data/","k", "./graphs/pso_third_data/", "pso_third")
    generate_graph_for_pso_dataset("./data/pso_third_data/","w", "./graphs/pso_third_data/", "pso_third")
    generate_graph_for_pso_dataset("./data/pso_third_data/","c1", "./graphs/pso_third_data/", "pso_third")
    generate_graph_for_pso_dataset("./data/pso_third_data/","c2", "./graphs/pso_third_data/", "pso_third")

def generate_graph_fourth_pso():
    generate_graph_for_pso_dataset("./data/pso_fourth_data/","p", "./graphs/pso_fourth_data/", "pso_fourth")
    generate_graph_for_pso_dataset("./data/pso_fourth_data/","k", "./graphs/pso_fourth_data/", "pso_fourth")
    generate_graph_for_pso_dataset("./data/pso_fourth_data/","w", "./graphs/pso_fourth_data/", "pso_fourth")
    generate_graph_for_pso_dataset("./data/pso_fourth_data/","c1", "./graphs/pso_fourth_data/", "pso_fourth")
    generate_graph_for_pso_dataset("./data/pso_fourth_data/","c2", "./graphs/pso_fourth_data/", "pso_fourth")

def generate_graph_initial_ga():
    generate_2d_graph("./data/ga_initial_data/", "It", "AveBestFitness", "./graphs/ga_initial_data/", "ga_initial")
    generate_graph_for_ga_dataset("./data/ga_initial_data/", "pmut_ind", "./graphs/ga_initial_data/", "ga_initial")
    generate_graph_for_ga_dataset("./data/ga_initial_data/", "pmut_gene", "./graphs/ga_initial_data/", "ga_initial")
    generate_graph_for_ga_dataset("./data/ga_initial_data/", "tournsize", "./graphs/ga_initial_data/", "ga_initial")
    generate_graph_for_ga_dataset("./data/ga_initial_data/", "indpb", "./graphs/ga_initial_data/", "ga_initial")

def generate_ga_second_graphs():
    generate_graph_for_ga_dataset("./data/ga_second_data/", "pmut_gene", "./graphs/ga_second_data/", "ga_second")
    generate_graph_for_ga_dataset("./data/ga_second_data/", "tournsize", "./graphs/ga_second_data/", "ga_second")
    generate_graph_for_ga_dataset("./data/ga_second_data/", "sigma", "./graphs/ga_second_data/", "ga_second")
    generate_graph_for_ga_dataset("./data/ga_second_data/", "indpb", "./graphs/ga_second_data/", "ga_second")
    generate_graph_for_ga_dataset("./data/ga_second_data/", "pcross", "./graphs/ga_second_data/", "ga_second")
    generate_graph_for_ga_dataset("./data/ga_second_data/", "pmut_ind", "./graphs/ga_second_data/", "ga_second")

def generate_ga_third_graphs():
    generate_graph_for_ga_dataset("./data/ga_third_data/", "tournsize", "./graphs/ga_third_data/", "ga_third")
    generate_graph_for_ga_dataset("./data/ga_third_data/", "pmut_ind", "./graphs/ga_third_data/", "ga_third")
    generate_graph_for_ga_dataset("./data/ga_third_data/", "pmut_gene", "./graphs/ga_third_data/", "ga_third")
    generate_graph_for_ga_dataset("./data/ga_third_data/", "indpb", "./graphs/ga_third_data/", "ga_third")
    generate_graph_for_ga_dataset("./data/ga_third_data/", "sigma", "./graphs/ga_third_data/", "ga_third")

def generate_ga_fourth_graphs():
    generate_graph_for_ga_dataset("./data/ga_fourth_data/", "tournsize", "./graphs/ga_fourth_data/", "ga_fourth")
    generate_graph_for_ga_dataset("./data/ga_fourth_data/", "pmut_ind", "./graphs/ga_fourth_data/", "ga_fourth")
    generate_graph_for_ga_dataset("./data/ga_fourth_data/", "pmut_gene", "./graphs/ga_fourth_data/", "ga_fourth")
    generate_graph_for_ga_dataset("./data/ga_fourth_data/", "indpb", "./graphs/ga_fourth_data/", "ga_fourth")
    generate_graph_for_ga_dataset("./data/ga_fourth_data/", "sigma", "./graphs/ga_fourth_data/", "ga_fourth")


def generate_ga_fifth_graphs():
    generate_graph_for_ga_dataset("./data/ga_fifth_data/", "tournsize", "./graphs/ga_fifth_data/", "ga_fifth")
    generate_graph_for_ga_dataset("./data/ga_fifth_data/", "pmut_ind", "./graphs/ga_fifth_data/", "ga_fifth")
    generate_graph_for_ga_dataset("./data/ga_fifth_data/", "pmut_gene", "./graphs/ga_fifth_data/", "ga_fifth")
    generate_graph_for_ga_dataset("./data/ga_fifth_data/", "indpb", "./graphs/ga_fifth_data/", "ga_fifth")
    generate_graph_for_ga_dataset("./data/ga_fifth_data/", "sigma", "./graphs/ga_fifth_data/", "ga_fifth")

def generate_ga_sixth_graphs():
    generate_graph_for_ga_dataset("./data/ga_sixth_data/", "tournsize", "./graphs/ga_sixth_data/", "ga_sixth")
    generate_graph_for_ga_dataset("./data/ga_sixth_data/", "pmut_ind", "./graphs/ga_sixth_data/", "ga_sixth")
    generate_graph_for_ga_dataset("./data/ga_sixth_data/", "pmut_gene", "./graphs/ga_sixth_data/", "ga_sixth")
    generate_graph_for_ga_dataset("./data/ga_sixth_data/", "indpb", "./graphs/ga_sixth_data/", "ga_sixth")
    generate_graph_for_ga_dataset("./data/ga_sixth_data/", "sigma", "./graphs/ga_sixth_data/", "ga_sixth")

def generate_final_graph_2D():
    generate_2d_graph("./data/ga_final_data/", "It", "AveBestFitness", "./graphs/ga_final_data/", "ga_final_data")

generate_graph_third_pso()
generate_graph_fourth_pso()