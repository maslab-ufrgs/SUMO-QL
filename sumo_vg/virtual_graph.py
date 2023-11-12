# functions to create virtual graph

import datetime as dt
import itertools
import math
import sys
from collections import Counter
from csv import DictReader
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Callable

import igraph as ig
import matplotlib.pyplot as plt
from pandas import DataFrame

# == Funções para as diferentes partes do programa ==


# - Importação de dados -


def is_num(num_str: str) -> bool:
    """
    Le csv inteiro e o guarda na memória, após, salva cada linha da leitura em uma lista
    verifica se string é um númerico ou não
    """
    try:
        float(num_str)
        return True
    except ValueError:
        return False


def should_include_line(row: dict[str, str], keys: list, link_as_vertex: bool) -> bool:
    """>
    Checks if line should be considered or discarded. This line will become a vertex of the virtual graph.
    Doesn't include line containing empty attribute, zero occupancy or border links (if graph will be composed of links).
    """
    # Checks if link is border link (only makes sense in grid network)
    is_border_link: Callable[[str], bool] = (
        lambda link: "top" in link
        or "bottom" in link
        or "right" in link
        or "left" in link
    )

    if link_as_vertex:
        if is_border_link(row["Link"]):
            return False
    if float(row["Occupancy"]) == 0 or any(row[key] == "" for key in keys):
        return False
    return True


def should_normalize(key_value: list) -> bool:
    """
    Determines if an imported column of the input csv should be normalized or not.
    """
    return is_num(str(key_value))


def csv_import(
    csv_path: str, first_id: int
) -> tuple[list, list]:  # recebe o caminho e o nome do arquivo a ser lido
    """
    Lê os dados do csv e retorna cada linha como um dicionário cujas keys são o header do csv e uma lista com as keys do dicionário
    """
    with open(csv_path) as file:
        # lê cada atributo como string, que será convertido posteriormente
        reading = DictReader(file)
        keys = list(reading.fieldnames or [])  # guardas as keys do dicionário

        attribute_name = keys[first_id - 1]
        link_as_vertex = attribute_name == "Link"

        lines = []
        id = 0  # identificador usado para criar arestas
        line_no = 2
        for line in reading:
            if should_include_line(line, keys, link_as_vertex):
                lines.append(
                    line
                )  # grava cada linha (um dicionário próprio) em uma lista de linhas
                line["id"] = id
                id += 1
            line_no += 1

        # converte atributos do dicionário para seus tipos respectivos (inicialmente são strings)
        for line in lines:
            for key in keys:
                if is_num(line[key]):
                    line[key] = float(
                        line[key]
                    )  # converte os atributos numéricos para float

    # retorna lista contendo dicionários e outra lista com as keys desses dicionários (são todas as mesmas)
    return lines, keys


def normalize_list(attributes: list) -> list:
    """
    Normaliza os valores de uma lista (com a fórmula n = (x-min)/(max-min)) e retorna uma lista com valores normalizados
    """
    minim = min(attributes)
    maxim = max(attributes)

    if (maxim - minim) == 0:
        exit("Erro na normalização: divisão por zero")

    return [(attribute - minim) / (maxim - minim) for attribute in attributes]


def normalize_dict(dicts: list, keys: list, labels: list) -> tuple[list, list]:
    """
    Normaliza os valores de uma lista de dicionários
    """
    normalized_dicts = dicts
    keys_norm = []
    for key in keys:
        if key not in labels and should_normalize(
            dicts[0][key]
        ):  # só os atributos que são númericos e não compõem o label serão normalizados
            attributes = [
                dic[key] for dic in normalized_dicts
            ]  # lista de atributos, usada para calcular fórmula da normalização
            attributes = normalize_list(attributes)

            for dic, norm_attribute in zip(normalized_dicts, attributes):
                # para cada dicionário na lista, atribui o valor respectivo da lista de atributos normalizados
                dic[f"{key} Norm"] = norm_attribute

            # monta lista com nomes dos atributos normalizados
            keys_norm.append(key + " Norm")
        else:
            keys_norm.append(key)

    # retorna a lista de dicionários normalizada
    return normalized_dicts, keys_norm


#  - Processamento da entrada -


def create_ids(dicts: list, used_attributes: list) -> list:
    """
    Recebe a lista contendo os vértices do grafo e os nomes dos atributos que irão compor o id, retornando a lista com os atributos concatenados
    """
    ids = []

    for node in dicts:
        attrib_names = []
        # para cada aributo, forma string que irá compor o nome do vértice
        for i in range(len(used_attributes)):
            if i == 0:  # o primeiro atributo do nome não recebe "_" antes
                attrib_names.append(f"{node[used_attributes[i]]}")
            else:
                attrib_names.append(f"_{node[used_attributes[i]]}")
        attrib_name = "".join(attrib_names)
        # monta a lista contendo os nomes dos vértices
        ids.append(attrib_name)

    return ids


def valid_ids(lista_ids: list) -> bool:
    """
    Recebe a lista contendo os ids dos nodos e retorna True se todos forem diferentes ou False caso contrário
    """
    id_pairs = itertools.combinations(lista_ids, 2)  # cria pares de ids

    return not any(pair[0] == pair[1] for pair in id_pairs)


def gen_attrib_list_str(attributes: list) -> str:
    """
    Recebe uma lista de atributos e monta uma string com estes atributos separados por "-"
    """
    return "".join(
        str(attributes[i]) if i == len(attributes) - 1 else f"{str(attributes[i])}-"
        for i in range(len(attributes))
    )


def get_file_name(directory_file: str) -> str:
    """
    Input: directory and name of file
    Output: str containing the name of the file
    """
    path = Path(directory_file)
    return path.stem if path.suffix == ".csv" else directory_file


def build_name(threshold: float, attributes: list, directory_file: str) -> str:
    """
    Recebe os parâmetros do usuário e gera o nome do arquivo que contém alguns dados do grafo
    """
    time = dt.datetime.now()
    current_hour = time.strftime("%H%M%S")
    attribute_list = gen_attrib_list_str(attributes)
    threshold_str = str(threshold)
    # remove o ponto do limiar para nao causar problemas com o nome e a extensão do arquivo
    proccessed_threshold_str = threshold_str.replace(".", "'")
    directory_file_stem = get_file_name(directory_file)

    return f"{current_hour}_atb{attribute_list}_l{proccessed_threshold_str}_{directory_file_stem}"


def convert_interval(interval: str) -> list:
    """
    Converte string representando intervalo numérico na forma "início-fim" em uma lista contendo os números naquele intervalo
    """
    nums = interval.split("-")

    begin = int(nums[0])
    end = int(nums[1])

    if begin > end:
        sys.exit("Erro: início do intervalo maior do que o final")

    interval_list = []

    x = begin
    while (
        x != end + 1
    ):  # preenche a lista incrementando os números do início do intervalo ao fim
        interval_list.append(x)
        x += 1

    return interval_list


def proccess_int_or_interval(entry: list | str) -> list:
    """
    Determina se a entrada é um intervalo numérico na forma "início-fim" ou uma lista de inteiros, retornando a lista de inteiros que corresponde ao intervalo ou a própria lista de inteiros
    """
    intervals = []

    for v in entry:
        if "-" in v:
            for num in convert_interval(v):
                intervals.append(num)
        else:
            intervals.append(int(v))  # transforma os números em inteiros

    return intervals


# - Criação de arestas -


def verify_edge(result_list: list, use_or_logic: bool) -> bool:
    """
    Dependendo da lógica escolhida, verifica se deve ser criada uma aresta entre um par de nodos
    """
    if use_or_logic:
        # dada a lista final de resultados, se houver algum verdadeiro, a aresta é criada
        return any(result == 1 for result in result_list)
    else:
        # dada a lista final de resultados, se houver algum falso, a aresta não é criada
        return not any(result == 0 for result in result_list)


def in_restriction(v1: dict, v2: dict, restriction_list: list | None) -> bool:
    """
    Dados dois vértices e uma lista de atributos usados como restrição, a função retorna -1 se a lista de restrições contiver "None",
    False se os vértices possuírem os mesmos valores para os mesmos atributos restritivos ou True se possuírem todos os valores diferentes
    para os mesmos atributos restritivos
    """
    if restriction_list is not None:
        return not any(
            v1[restriction] == v2[restriction] for restriction in restriction_list
        )
    return True


def in_threshold(
    v1: dict, v2: dict, attribute: str, threshold: float, precision: int
) -> bool:
    """
    Verifica se um atributo entre dois dicionários está dentro do limiar ou não
    recebe dois dicionários, um atributo, um limiar e a precisão da diferença entre atributos
    """
    getcontext().prec = precision
    return abs(Decimal(v1[attribute]) - Decimal(v2[attribute])) <= threshold


def build_edges(
    attributes: list,
    dicts: list,
    restrictions: list | None,
    use_or_logic: bool,
    threshold: float,
    precision: int,
):
    """
    Monta uma lista de arestas a partir de uma lista de atributos, uma de dicionários, uma de restrições, uma lógica para montar arestas e um limiar
    """
    edges = []
    edges_wheights = []
    # para cada par de dicionários da lista
    for v1, v2 in itertools.combinations(dicts, 2):
        # indica se para cada atributo, deve haver uma aresta (1) ou não (0)
        results_list = []
        if in_restriction(v1, v2, restrictions):
            for attribute in attributes:
                # se a diferença absoluta do valor do atributo de dois nodos for menor ou igual ao limiar
                if in_threshold(v1, v2, attribute, threshold, precision):
                    # lista de resultados para aquele par contém 1, isto é, verdadeiro
                    results_list.append(1)
                else:
                    results_list.append(0)  # caso contrário contém zero

            if verify_edge(results_list, use_or_logic):
                # adiciona a aresta à lista de arestas, utilizando o identificador numérico dos vértices
                edges.append((v1["id"], v2["id"]))
                # como, para cada atributo, a lista contém 1 se há aresta e 0 se não há, a soma desses uns dará o número de atributos dentro do limiar entre o par de nodos
                edges_wheights.append(sum(results_list))

    # retorna lista com arestas e lista com pesos das arestas
    return edges, edges_wheights


# - Toma medidas sobre o grafo -


def has_constly_measure(measures: list | None, costly_measures: list) -> bool:
    """
    Determina se lista de medidas possui alguma medida considerada custosa
    """
    if measures is not None:
        return any(costly_measure in measures for costly_measure in costly_measures)
    return False


def calc_measures(graph: ig.Graph, measures: list) -> dict:
    """
    Calcula medidas de centralidade do grafo, retornando um dicionário com as medidas
    """
    measures_values: dict[str, list] = {"label": graph.vs["label"]}

    for measure in measures:
        measures_values[measure] = list(
            map(lambda x: round(x, 4), getattr(graph, measure)())
        )

    return measures_values


def calculate_frequency_keys(graph: ig.Graph, attribute: str) -> dict:
    """
    Input: graph representing network composed of nodes that are dictionaries
    Output: dictionary with the frequency that the specified key appears in the graph
    """
    list_keys = [v[attribute] for v in graph.vs]

    dict_freq_keys = dict(Counter(key for key in list_keys))

    # creates sorted dictionary by value in key in decreasing order
    sorted_dict_freq_keys = {
        item[0]: item[1]
        for item in sorted(dict_freq_keys.items(), key=lambda x: (-x[1], x[0]))
    }

    processed_dict = dict()
    processed_dict[attribute] = sorted_dict_freq_keys.keys()

    frequency_list = [item[1] for item in list(sorted_dict_freq_keys.items())]

    processed_dict["frequency"] = frequency_list

    return processed_dict


def build_table(data: dict, name: str, type: str) -> None:
    """
    Recebe os dados em uma lista, o nome que o arquivo de saída terá e se o dados se referem
    às medidas de centralidade ("centrality") ou frequência ("frequency")
    """
    local_path = Path(f"results/tables/{type}")
    local_path.mkdir(exist_ok=True, parents=True)
    local_nome = Path(name)

    _, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.axis("tight")

    columns = list(data.keys())
    df = DataFrame(columns, columns=columns)
    print(columns)
    print(df.head())
    df_headers = df.columns.tolist()
    df = df.sort_values(df_headers[1], ascending=False)

    ax.table(
        cellText=df.values,
        colLabels=list(df.columns),
        cellLoc="center",
        loc="upper center",
    )

    plt.savefig(f"{str(local_path/local_nome)}", format="pdf", bbox_inches="tight")

    print(f"Table '{local_nome}' generated at {local_path}")


# - Visual do grafo -


def bbox_calc(n_edges: int) -> tuple:
    """
    Calcula o tamanho da imagem do grafo
    """
    bbox_limit = 3000

    # criada com raiz quadrada, chegando perto dos pontos
    bbox = math.sqrt(n_edges) * 150

    if bbox > bbox_limit:  # define limite para bbox
        bbox = bbox_limit

    return (bbox, bbox)


def determ_vertex_color(degree: float, avg: float) -> str:
    """
    Determina cor de um vértice, quão maior o degree, mais quente a cor
    """
    razao = degree / avg

    if razao < 0.1:
        return "#ADD8E6"  # light blue
    elif razao < 0.3:
        return "#0000CD"  # medium blue
    elif razao < 0.5:
        return "#0000FF"  # blue
    elif razao < 0.7:
        return "#90EE90"  # light green
    elif razao < 0.9:
        return "#00FF00"  # green
    elif razao < 1.1:
        return "#006400"  # dark green
    elif razao < 1.3:
        return "#FFFF00"  # yellow
    elif razao < 1.5:
        return "#FFCC00"  # dark yellow
    elif razao < 1.7:
        return "#FFA500"  # orange
    elif razao < 1.9:
        return "#FF8C00"  # dark orange
    elif razao < 2.1:
        return "#FF4500"  # orange red
    elif razao < 2.3:
        return "#FF3333"  # light red
    elif razao < 2.5:
        return "#FF0000"  # red
    elif razao < 2.7:
        return "#8B0000"  # dark red
    elif razao < 2.9:
        return "#9370DB"  # medium purple
    elif razao < 3.1:
        return "#A020F0"  # purple
    else:
        return "#000000"  # black


def color_list(g: ig.Graph) -> list:
    """
    Cria lista de cores dos vértices do grafo
    """
    degrees = g.degree()

    avg_degree = 0 if len(degrees) == 0 else sum(degrees) / len(degrees)

    colors = []
    for degree in degrees:
        colors.append(determ_vertex_color(degree, avg_degree))

    return colors


def determine_visual_style(g: ig.Graph) -> dict:
    """
    Determina características visuais do grafo
    """
    font_lower_bound = 8
    visual_style = {}
    visual_style["bbox"] = bbox_calc(g.ecount())
    visual_style["margin"] = 60
    visual_style["edge_color"] = "grey"
    visual_style["vertex_color"] = color_list(g)
    visual_style["vertex_label_dist"] = 1.1
    # tamanho do vértice será o maior número entre 15 e 6*math.sqrt(d)
    visual_style["vertex_size"] = [max(15, 6 * math.sqrt(d)) for d in g.degree()]
    # tamanho da fonte entre 10 e 8, dependendo do grau do vértice
    visual_style["vertex_label_size"] = [
        max(
            10,
            4 * math.sqrt(d)
            if 4 * math.sqrt(d) < font_lower_bound
            else font_lower_bound,
        )
        for d in g.degree()
    ]
    visual_style["layout"] = g.layout("auto")

    return visual_style


def has_min_degree(graph: ig.Graph, min_d: float) -> bool:
    """
    Determina se o grafo possui algum vértice com grau mínimo passado
    """
    return any(v.degree() < min_d for v in graph.vs)


def calc_max_step(dicts: list, keys: list) -> int:
    """
    Calcula step máximo
    """
    step_name = list(filter(lambda x: x == "Step" or x == "step", keys))[0]

    step_list = [v[step_name] for v in dicts]

    return int(max(step_list))


def is_within_interval(
    inf_threshold: int, sup_threshold: int, value: float, last_interval: bool
) -> bool:
    """
    Recebe número e verifica se este está dentro de um intervalo [x, y) (ou, se for o último intervalo, [w, z])
    """
    if not last_interval:
        return value >= inf_threshold and value < sup_threshold
    else:
        return value >= inf_threshold and value <= sup_threshold


def neighbors(list_attribute_vertices: list) -> list:
    """
    Recebe uma lista de vértices com mesmo atributo especificado e retorna uma lista contendo todos os vizinhos desse atributo
    """
    list_attribute_neighbors = []
    for attribute_v in list_attribute_vertices:
        list_vertex_neighbors = attribute_v.neighbors()
        for neighbor_v in list_vertex_neighbors:
            if neighbor_v not in list_attribute_neighbors:
                list_attribute_neighbors.append(neighbor_v)

    return list_attribute_neighbors


def neighbors_within_interval(
    neighbors: list,
    interval: tuple,
    last_interval: bool,
    step_name: str,
    attribute_name: str,
) -> list:
    """
    Recebe uma lista de vizinhos do atributo especificado e filtra os vértices que estão no intervalo passado
    """
    inf_threshold = interval[0]
    sup_threshold = interval[1]

    # para cada vertice, verificar se esta no intervalo e adicionar à lista
    neighbors_within = set(
        neighbor[attribute_name]
        for neighbor in neighbors
        if is_within_interval(
            inf_threshold, sup_threshold, neighbor[step_name], last_interval
        )
    )

    return list(neighbors_within)


def build_neighbors(
    graph: ig.Graph, keys: list, attribute_name: str, interval: int, max_step: int
) -> dict:
    """
    Cria um dicionário contendo todos os vizinhos em determinado intervalo de tempo do grafo agrupados por um atributo (ex. link ou junction).
    """
    neighbors = dict()
    step_name = list(filter(lambda x: x == "Step" or x == "step", keys))[0]

    list_attributes = [v[attribute_name] for v in graph.vs]

    unique_attributes_list = list(set(attribute for attribute in list_attributes))

    # constroi lista de intervalos (compostos por uma tupla) utilizando o último step e o tamanho do intervalo selecionado: ["[0-1)", "[1-2) ... [n-1, n]"]
    intervals = []
    # obs: intervalo é fechado à esquerda e aberto à direita, mas o último intervalo é fechado nos dois lados
    interval_name = (0, 0)
    inf_threshold = 0
    sup_threshold = 0
    if interval == 0:
        exit("Erro, intervalo não pode ser zero")
    interval_num = int(max_step / interval)
    for i in range(interval_num):
        sup_threshold += interval
        interval_name = (inf_threshold, sup_threshold)
        inf_threshold += interval
        intervals.append(interval_name)

    for attribute in unique_attributes_list:
        neighbors[attribute] = dict()

        # lista de vértices do grafo com atributo especificado
        list_attribute_vertices = list(
            filter(lambda v: v[attribute_name] == attribute, graph.vs)  # type: ignore
        )

        for i in range(interval_num):
            # se for o último intervalo, ele será fechado dos dois lados
            last_interval = i == interval_num - 1

            lower_interval = intervals[i][0]
            upper_interval = intervals[i][1]

            neighbors[attribute][
                intervals[i]
            ] = []  # empties list of neighbors for that interval
            v_neighbors_at_interval = []
            for v_attribute in list_attribute_vertices:
                if is_within_interval(
                    lower_interval,
                    upper_interval,
                    v_attribute[step_name],
                    last_interval,
                ):
                    v_neighbors_at_interval = (
                        v_attribute.neighbors()
                    )  # gets neighbors at interval
                for v_neighbor in v_neighbors_at_interval:
                    if (
                        v_neighbor[attribute_name]
                        not in neighbors[attribute][intervals[i]]
                    ):
                        neighbors[attribute][intervals[i]].append(
                            v_neighbor[attribute_name]
                        )  # filters and appends list

    return neighbors


def generate_graph_neighbors_dict(
    csv_name: str,
    numeric_attribs: list,
    numeric_labels: list,
    numeric_restrictions: list | None,
    threshold: float,
    use_or_logic: bool,
    measures: list | None,
    no_image: bool,
    raw_graph: bool,
    giant_component: bool,
    not_normalize: bool,
    min_degree: int,
    min_step: int,
    costly_edges: int,
    precision: int,
    neighbors_interval: int,
    network_name: str,
) -> dict:
    """
    Main script to generate the virtual graph, its image, take centrality measurements of it and generate
    finally the virtual graph neighbors dictionary
    """

    # == Processa listas numéricas ==

    if numeric_attribs != ["ALL"]:
        numeric_attribs = proccess_int_or_interval(numeric_attribs)

    numeric_labels = proccess_int_or_interval(numeric_labels)

    if numeric_restrictions is not None:
        numeric_restrictions = proccess_int_or_interval(numeric_restrictions)

    # == Lê csv, traduz entrada numérica dos ids para atributos e normaliza dados, se foi pedido ==

    print("Reading file...")
    dicts, keys = csv_import(csv_name, numeric_labels[0])

    id_labels = (
        []
    )  # usada como label do grafo, indica também atributos que não serão normalizados
    for (
        num_id
    ) in (
        numeric_labels
    ):  # traduz os número passados como argumento correspondente às colunas
        id_labels.append(
            keys[num_id - 1]
        )  # numeração das colunas começa em 1, por isso -1

    if not not_normalize:
        dicts, keys = normalize_dict(dicts, keys, id_labels)
    print("File read.")

    # Traduz entrada numérica dos outros parâmetros

    print("Translating atributes...")

    attributes = []
    if numeric_attribs != ["ALL"]:
        for num_atb in numeric_attribs:
            attributes.append(keys[num_atb - 1])
    else:
        for attribute in keys:
            if attribute not in id_labels:
                attributes.append(
                    attribute
                )  # atributos usados serão todos menos os que compõem o id

    if numeric_restrictions is not None:
        restrictions = [keys[num_rest - 1] for num_rest in numeric_restrictions]
    else:
        restrictions = numeric_restrictions

    vertex_attribute_name = keys[numeric_labels[0] - 1]

    # == Prints para mostrar parâmetros selecionados ==

    print(f"Attributes: {attributes}")
    print(f"Labels: {id_labels}")
    print(f"Restrictions: {restrictions}")
    print(f"Centrality measures: {measures}")
    output_m = "True" if measures is not None else "False"
    print(f"Take centrality measures: {output_m}")
    print(f"Limiar: {threshold}")
    print(f"Use or logic: {use_or_logic}")
    print(f"File: {csv_name}")
    print(f"No virtual graph image: {no_image}")
    print(f"Use pure virtual graph: {raw_graph}")
    print(f"Only plot giant component: {giant_component}")
    print(f"Don't normalize input: {not_normalize}")
    print(f"Plots vertices with a degree bigger or equal to: {min_degree}")
    print(f"Plots vertices with a step bigger or equal to: {min_step}")
    print(
        f"Amplitude of timestep of virtual graph neighbors dictionary: {neighbors_interval} steps"
    )
    print(f"Virtual graph's vertices: {vertex_attribute_name}")

    # == Cria ids ==

    print("Generating labels...")
    id_list = create_ids(
        dicts, id_labels
    )  # monta lista de identificadores dos vértices do grafo
    if not valid_ids(id_list):  # se os ids gerados não forem únicos
        print("Error! Labels created aren't unique. Use other atributes")
        sys.exit("Exiting program")
    else:
        print("Labels are valid")

    # == Monta lista de arestas ==

    print("Generating edges...")
    edges, edge_weights = build_edges(
        attributes, dicts, restrictions, use_or_logic, threshold, precision
    )

    # == Cria grafo e o processa ==

    print("Atributing values to the virtual graph...")
    g_raw = ig.Graph()
    n_vertices = len(dicts)
    g_raw.add_vertices(n_vertices)
    g_raw.vs["label"] = id_list  # label do grafo é a lista de ids
    g_raw.add_edges(edges)  # grafo recebe as arestas
    g_raw.es["peso"] = edge_weights  # arestas recebem seus pesos

    for key in keys:
        g_raw.vs[key] = [
            veiculo[key] for veiculo in dicts
        ]  # grafo recebe os atributos dos dicionários

    # pega o nome do atributo referente ao step no arquivo de entrada
    step_names = list(filter(lambda x: x == "Step" or x == "step", keys))[0]

    g = g_raw.copy()  # copia o grafo original
    to_delete_vertices = []
    # verifica se o usuário escolheu remover os vértices cujo grau é zero
    if not raw_graph:
        if (
            min_step > 0
        ):  # se for considerado um step mínimo, remover vértices abaixo desse step
            for v in g.vs:
                if v[step_names] < min_step:
                    to_delete_vertices.append(
                        v
                    )  # seleciona ids com step abaixo do mínimo
            g.delete_vertices(
                to_delete_vertices
            )  # remove vértices com step abaixo do mínimo

        to_delete_vertices = []  # remove vértices cujo grau é zero
        for v in g.vs:
            if v.degree() == 0:
                to_delete_vertices.append(v)  # seleciona ids com degree zero
        g.delete_vertices(
            to_delete_vertices
        )  # remove todos os ids que não formam nenhuma aresta (cujo grau é zero)
    else:
        if (
            min_step > 0
        ):  # se o usuário escolheu não remover os vértices de grau zero, verificar se escolheu um step mínimo
            for v in g.vs:
                if v[step_names] < min_step:
                    to_delete_vertices.append(v)
            g.delete_vertices(to_delete_vertices)

    print("Done")  # finalizou a atribuição

    # mostra informações do grafo, como número de vértices e quantidade de arestas
    print("Information about the virtual graph:")
    print(g.degree_distribution())
    print(g.summary())

    # == Trata custo computacional ==

    # lista de medidas que são custosas e não desejáveis de ser tomadas se o grafo for muito grande
    costly_measures = ["betweenness"]
    has_costly_measure = has_constly_measure(measures, costly_measures)
    new_measure_list = (
        measures.copy() if measures is not None else []
    )  # usada para, se for escolhido, filtrar as medidas que são custosas
    cost = 0  # define a intensidade do custo computacional: 0 para baixo, 1 para médio e 2 para alto
    big_graph_option = 0
    data_name = ""

    if g.ecount() <= costly_edges:
        cost = 0
    else:
        cost = 1

    # se o usuário optou por gerar uma imagem do grafo ou realizar alguma medida
    if not no_image or measures is not None:
        if cost == 1:
            if has_costly_measure:
                print(
                    f"The graph has more than {costly_edges} edges. Do you really wish to generate an image of this graph and take costly centrality measures?"
                )
                print("1 - Take measures and generate image")
                print("2 - Only take measures")
                print("3 - Only genrate image")
                print("4 - Don't generate image neither take measures")

                # recebe o input do usuário, verificando a consistência da entrada
                while (
                    big_graph_option != 1
                    and big_graph_option != 2
                    and big_graph_option != 3
                    and big_graph_option != 4
                ):
                    big_graph_option = int(input("Type your option: "))
                    if (
                        big_graph_option != 1
                        and big_graph_option != 2
                        and big_graph_option != 3
                        and big_graph_option != 4
                    ):
                        print("Invalid option, type again.")
            else:
                print(
                    f"The graph has more than {costly_edges} edges. Do you really wish to generate an image of this graph?"
                )
                print("1 - Yes")
                print("2 - No")

                # recebe o input do usuário, verificando a consistência da entrada
                while big_graph_option != 1 and big_graph_option != 2:
                    big_graph_option = int(input("Type your option: "))
                    if big_graph_option != 1 and big_graph_option != 2:
                        print("Invalid option, type again.")

            # se for escolhido para não tomar medidas custosas
            if big_graph_option == 3 or big_graph_option == 4:
                new_measure_list = (
                    [measure for measure in measures if measure not in costly_measures]
                    if measures is not None
                    else []
                )

        # salva informações que irão compor os nomes dos arquivos de saída
        data_name = build_name(threshold, numeric_attribs, network_name)

    # == Gera imagem do grafo ==

    if not no_image:
        # se foi selecionado para fazer a imagem do grafo, ou se não for custoso
        if big_graph_option == 1 or big_graph_option == 3 or cost == 0:
            print("Ploting virtual graph...")
            if g.vcount() != 0:
                vg_path = Path("results/graphs")
                vg_path.mkdir(exist_ok=True, parents=True)
                vg_name = Path(f"img_{data_name}.pdf")

                if (
                    giant_component
                ):  # se foi escolhido para apenas mostrar o giant component do grafo
                    g_plot = g.components().giant().copy()
                else:  # caso contrario, mostrar todo o grafo
                    g_plot = g.copy()

                if min_degree < 0:
                    print("Minimum degree is negative and will not be considered")
                elif min_degree != 0:
                    while has_min_degree(g_plot, min_degree):
                        to_delete_vertices = []
                        for v in g_plot.vs:
                            if (
                                v.degree() < min_degree
                            ):  # apenas deixar vértices cujo grau é igual ou maior que o passado em mdeg
                                to_delete_vertices.append(v)
                        g_plot.delete_vertices(to_delete_vertices)

                if g_plot.vcount() != 0:  # se o grafo não estiver vazio, plotar
                    visual_style = determine_visual_style(g_plot)
                    ig.plot(g_plot, target=str(vg_path / vg_name), **visual_style)
                    print(f"Image '{str(vg_name)}' generated at {vg_path}")
                else:
                    print("Empty virtual graph, no image generated")
            else:
                print("Empty virtual graph, no image generated")
        else:
            print("The virtual graph will not be ploted")

    # == Toma medidas de caracterização ==

    if len(new_measure_list) > 0:
        print("Generating table...")

        if len(new_measure_list) != 0:
            if g.vcount() != 0:
                table_name = f"table_{data_name}.pdf"
                freq_table_name = f"freq_table_{data_name}.pdf"
                # tabela com as medidas de caracterização selecionadas é gerada
                build_table(
                    data=calc_measures(g, new_measure_list),
                    name=table_name,
                    type="centrality",
                )
                # tabela de frequências é gerada
                build_table(
                    data=calculate_frequency_keys(g, attribute=vertex_attribute_name),
                    name=freq_table_name,
                    type="frequency",
                )
            else:
                print("Empty graph, no table generated")
        else:
            print("Centrality measurements list is empty")

    neighbors = build_neighbors(
        g,
        keys,
        vertex_attribute_name,
        neighbors_interval,
        max_step=calc_max_step(dicts, keys),
    )

    print("")

    return neighbors
