# standalone script to generate virtual graph dictionary file

import argparse as ap
import pickle
import sys
import time
from pathlib import Path

import igraph as ig

from sumo_vg.virtual_graph import *

# == Variáveis globais ==
costly_n_edges = 2000  # quantidade de arestas para que o grafo seja considerado custoso
precision = 10  # precisao do resultado da subtração de atributos dos vértices para comparação com limiar


def main():
    start_time = time.time()  # inicia temporizador de saída

    # == Argparse e argumentos do usuário ==

    parser = ap.ArgumentParser()

    vg_args_creation = parser.add_argument_group("Creating Virtual Graph")
    vg_args_adjusting = parser.add_argument_group("Adjusting Virtual Graph")
    vg_args_plotting = parser.add_argument_group("Plotting Virtual Graph")
    vg_args_measuring = parser.add_argument_group("Measuring Virtual Graph")

    vg_args_creation.add_argument(
        "-f",
        "--vg-file",
        dest="vg_file",
        help="Path to csv file that will be used as input for the virtual graph.",
    )

    vg_args_creation.add_argument(
        "-atb",
        "--vg-attributes",
        dest="vg_attributes",
        default=["ALL"],
        nargs="+",
        help="List of attributes used to create the virtual graph. Attribute is given by the number of the column "
        "of the input csv. (default = ['ALL'])",
    )

    vg_args_creation.add_argument(
        "-id",
        "--vg-label",
        dest="vg_label",
        nargs="+",
        help="List of attributes that will compose the label of each vertex in the virtual graph. Attribute is given "
        "by the number of the column of the input csv. The first attribute passed will determine which attribute is used "
        "to aggregate the virtual graph neighbors, i.e. aggregate by link or junction.",
    )

    vg_args_creation.add_argument(
        "-rst",
        "--vg-restrictions",
        dest="vg_restrictions",
        default=[],
        nargs="+",
        help="List of attributes that the vertices cannot share in order to create an edge in the virtual graph. Attribute"
        " is given by the number of the column of the input csv. (default = [])",
    )

    vg_args_creation.add_argument(
        "-tsh",
        "--vg-threshold",
        dest="vg_threshold",
        type=float,
        default=0,
        help="Threshold used to create an edge in the virtual graph. (default = 0)",
    )

    vg_args_adjusting.add_argument(
        "-or",
        "--use-or-logic",
        dest="use_or_logic",
        action="store_true",
        default=False,
        help="Flag that indicates or logic instead of the and logic to create an edge between vertices given multiple "
        "attributes. (default = false)",
    )

    vg_args_measuring.add_argument(
        "-ms",
        "--centrality-measures",
        dest="centrality_measures",
        default=[],
        nargs="+",
        help="List of centrality measures to be taken of the virtual graph. (default = [])",
    )

    vg_args_plotting.add_argument(
        "-ni",
        "--no-image",
        dest="no_image",
        action="store_true",
        default=False,
        help="Flag to indicate to the script not to generate the virtual graph image. (default = false)",
    )

    vg_args_adjusting.add_argument(
        "-rgraph",
        "--raw-graph",
        dest="raw_graph",
        action="store_true",
        default=False,
        help="Flag to indicate not to remove vertices with degree zero. (default = false)",
    )

    vg_args_plotting.add_argument(
        "-giant",
        "--giant-component",
        dest="giant_component",
        action="store_true",
        default=False,
        help="Flag to indicate that only the giant component of the virtual graph should be presented in "
        "its image. (default = false)",
    )

    vg_args_adjusting.add_argument(
        "-not-norm",
        "--vg-not-normalize",
        dest="vg_not_normalize",
        action="store_true",
        default=False,
        help="Flag to indicate to the script not to normalize the input csv data to generate the virtual"
        " graph. (default = false)",
    )

    vg_args_plotting.add_argument(
        "-mdeg",
        "--min-degree",
        dest="min_degree",
        type=int,
        default=0,
        help="Determines the minimum degree a vertex should have in order to be plotted in the virtual graph "
        "image. (default = 0)",
    )

    vg_args_plotting.add_argument(
        "-mstep",
        "--vg-min-step",
        dest="vg_min_step",
        type=int,
        default=0,
        help="Determines the minimum step a vertex should have in order to be plotted in the virtual graph image. "
        "(default = 0)",
    )

    vg_args_adjusting.add_argument(
        "-int",
        "--interval",
        type=int,
        default=250,
        help="Amplitude of the timestep interval of the virtual graph neighbors dictionary. (default = 250)",
    )

    args = parser.parse_args()

    csv_name = args.vg_file  # nome do arquivo csv a ser usado para gerar grafo
    numeric_attribs = (
        args.vg_attributes
    )  # lista de atributos, passados como número da coluna
    numeric_labels = (
        args.vg_label
    )  # lista de ids usados no label, passados como número da coluna
    numeric_restrictions = (
        args.vg_restrictions
    )  # lista de restricoes para criar arestas, passadas como número da coluna
    threshold = args.vg_threshold  # limiar usado para criar arestas
    use_or_logic = args.use_or_logic  # lógica para criar arestas
    measures = args.centrality_measures  # lista de medidas que serão tomadas do grafo
    no_image = args.no_image  # define se será gerada uma imagem do grafo ou não
    # define se será usado o grafo sem processamento (remover vértices de grau zero) ou não
    raw_graph = args.raw_graph
    giant_component = (
        args.giant_component
    )  # define se apenas o giant component será mostrado na imagem
    not_normalize = (
        args.vg_not_normalize
    )  # define se os dados usados serão normalizados
    min_degree = (
        args.min_degree
    )  # apenas serão mostrados vértices com grau a partir do especificado
    min_step = (
        args.vg_min_step
    )  # apenas serão considerados vértices cujo step é maior ou igual a este valor
    amplitude_interval = (
        args.interval
    )  # amplitude of the virtual graph neighbors dictionary

    # == Verfica consistência de entrada ==

    if args.vg_label is None:
        print("Error! Labels parameter wasn't informed!")
        sys.exit("Exiting program")

    if args.vg_file is None:
        print("Error! Input file path and name wasn't informed")
        sys.exit("Exiting program")

    print("Parameters OK")  # se todos os parâmetros necessário foram informados

    vg_neighbors_dict = generate_graph_neighbors_dict(
        csv_name=csv_name,
        numeric_attribs=numeric_attribs,
        numeric_labels=numeric_labels,
        numeric_restrictions=numeric_restrictions,
        threshold=threshold,
        use_or_logic=use_or_logic,
        measures=measures,
        no_image=no_image,
        raw_graph=raw_graph,
        giant_component=giant_component,
        not_normalize=not_normalize,
        min_degree=min_degree,
        min_step=min_step,
        costly_edges=2000,
        precision=10,
        neighbors_interval=amplitude_interval,
        network_name=get_file_name(csv_name),
    )

    # Saves dictionary to pickle file
    dict_path = Path("results/dictionaries")
    dict_path.mkdir(exist_ok=True, parents=True)
    dict_pickle_file_name = Path(f"dict_{get_file_name(csv_name)}.pkl")
    with open(dict_path / dict_pickle_file_name, "wb") as dict_pickle_file:
        pickle.dump(vg_neighbors_dict, dict_pickle_file)
    print(f"Generated dict file '{str(dict_pickle_file_name)}' at {str(dict_path)}")

    elapsed_time = time.time() - start_time  # Temporizador de saída
    print(f"Finished in {elapsed_time:.4f} seconds\n")


if __name__ == "__main__":
    main()
