import pickle
import re
from collections import Counter
from pathlib import Path
import dash_bootstrap_components as dbc
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spacy
from dash import Dash, ALL, ctx, dcc, html, Input, Output, State
from dash.dash_table import DataTable
from itertools import combinations
from plotly.subplots import make_subplots
import dash_ag_grid as dag


# ---- PLATFORM ----

nlp = spacy.load("en_core_web_sm")

G = nx.Graph()

tokens_changed: bool = False
stopped_words = set()
unstopped_words = set()
assigned_codes = dict()
active_data = list()
has_generated = False

theoretical_code_list = [ 
    "emergent",
    "centralized",
    "probabilistic",
    "deterministic",
    "feedback",
    "fitting",
    "levels",
    "slippage",
]

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)


# ---- NLP ----


def parse_raw_text(txt: str, timestamp=False, is_interviewer=False):

    global tokens_changed

    data = list()

    # parse the text
    input_lines = [
        line.strip().replace("\n", "")
        for line in txt.splitlines()
        if len(line.strip()) > 0 and line.count(":") > 2
    ]

    # to parse the text line by line
    re_time_splitter = re.compile(r"(\[[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\])")

    if not is_interviewer:
        input_lines = [
            line for line in input_lines if line.lower().count("interviewer") == 0
        ]

    for i, line in enumerate(input_lines):
        # cleans
        _, time, speaker_speech = re_time_splitter.split(line)
        speaker, utterance = speaker_speech.strip().split(":", maxsplit=1)
        speaker = str(speaker).strip()

        row = {"line": i + 1}

        row['include?'] = True

        if timestamp:
            row["time"] = time[1:-1]

        if speaker:
            row["speaker"] = speaker

        # here would I go through and make each token bold using markdown?
        row["utterance"] = utterance.strip()

        data.append(row)

        if i not in assigned_codes.keys():
            assigned_codes[i] = [False] * len(theoretical_code_list) # initializing the assigned_codes dictionary 

    tokens_changed = True

    return data


def generate_code_checkboxes(line_num, values=None):

    if values is not None:
        assigned_codes[line_num] = values

    container = html.Div(
        [
            html.Div(
                [
                    dbc.Checkbox(
                        label=code[1],
                        value=assigned_codes[line_num][code[0]],
                        id={"type": "code-checkbox", "index": code[1]},
                    )
                ],
                className="w-50",
            )
            for code in enumerate(theoretical_codes_list)
        ],
        className="d-flex align-content-start flex-wrap",
        id="code-checkboxes-container",
    )
    return container

# mapping use of certain "tokens" --> words?
def process_utterance(raw_text):

    global nlp

    doc = nlp(raw_text.strip().lower())

    all_tokens = [
        token.lemma_
        for token in doc
        if not nlp.vocab[token.lemma].is_stop and not token.is_punct
    ]

    token_counts = Counter(all_tokens)

    data_dict = {
        "token": list(token_counts.keys()),
        "count": list(token_counts.values()),
    }

    df = pd.DataFrame.from_dict(data_dict)

    buttons_for_text = html.Div(
        [
            html.Span(
                dbc.Button(
                    token.text,
                    id={"type": "toggle-token", "index": token.lemma_, "stop": True},
                    n_clicks=0,
                    color="light",
                    class_name="m-1",
                    size="sm",
                )
            )
            if nlp.vocab[token.lemma].is_stop
            else html.Span(token.text, className="mx-1")
            if nlp.vocab[token.lemma].is_punct
            else html.Span(
                dbc.Button(
                    token.text,
                    id={"type": "toggle-token", "index": token.lemma_, "stop": False},
                    n_clicks=0,
                    color="warning",
                    class_name="m-1",
                    size="sm",
                )
            )
            for token in doc
        ]
    )

    # why a treemap?
    fig = px.treemap(
        df,
        path=[px.Constant("tokens"), "token"],
        values="count",
        color="count",
        hover_data="token",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=df["count"].mean(),
    )

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    fig.update_coloraxes(showscale=False)

    token_treemap = dcc.Graph(figure=fig, responsive=True, style={"height": "200px"})

    return buttons_for_text, token_treemap

def pickle_model(mode_name):
    global nlp

    models_folder = Path("./models/")
    models_folder.mkdir(exist_ok=True)

    mode_folder = models_folder / mode_name
    mode_folder.mkdir(exist_ok=True)

    stopwords_file = mode_folder / "stopwords.pickle"
    theoretical_codes_file = mode_folder / "theoretical_codes.pickle"

    with open(stopwords_file, "wb") as swf:
        pickle.dump(
            (stopped_words, unstopped_words), swf, protocol=pickle.HIGHEST_PROTOCOL
        )

    with open(theoretical_codes_file, "wb") as tcf:
        pickle.dump(assigned_codes, tcf, protocol=pickle.HIGHEST_PROTOCOL)


def generate_knowledge_graph(start, end, sentence_boost=False, with_interviewer=False):
    global nlp
    global active_data

    new_G = nx.Graph()

    # if showing a cumulative graph (start == 0), generate nodes for just until that point
    #    otherwise, generate nodes for the entire transcript
    data_dict_list = active_data[0:end] if start == 0 else active_data

    for line in data_dict_list:
        if ((with_interviewer or (not with_interviewer and line["speaker"].lower() != "interviewer"))
            and line['include?']):
            doc_line = nlp(line["utterance"].strip().lower()) # cleans

            tokens = [t.lemma for t in doc_line if not t.is_punct and not t.is_stop and not nlp.vocab[t.lemma].is_stop] # cleans

            token_counts = Counter(tokens)
            unique_tokens = list(token_counts.keys())

            for t in unique_tokens:
                if new_G.has_node(t):
                    new_G.nodes[t]["count"] += token_counts[t]
                else:
                    new_G.add_node(t, count=token_counts[t], label=nlp.vocab.strings[t])

            for t1, t2 in combinations(unique_tokens, 2):
                if new_G.has_edge(t1, t2):
                    new_G[t1][t2]["weight"] += 1
                else:
                    new_G.add_edge(t1, t2, weight=1)

            # boost edges between tokens within the same sentences by 1
            #   if there are more than 1 sentences in the line --> is this measuring the "amount" of talking?

            if sentence_boost:
                sentences = [s for s in doc_line.sents]

                if len(sentences) > 1:
                    for s in sentences:
                        sent_tokens = [
                            t.lemma for t in s if not t.is_punct and not t.is_stop
                        ]

                        # unique tokens within the sentence
                        s_tokens = set(sent_tokens)

                        for t1, t2 in combinations(s_tokens, 2):
                            new_G[t1][t2]["weight"] += 1

    return new_G


def display_knowledge_graph(
    start_line=0,  # if > 0, dmc mode is activated
    end_line=1,
    case_name="",
    raw_frequency=True,
    with_codes=False,
    layout=1,
    spring_iterations=30,
    spring_k=0.2,
    min_co_occurrence=1,
    min_dmc_co_occurrence=2,
    size_multiplier=2,
    show_interviewer=False
):
    global nlp
    global G
    global tokens_changed

    # UA > if any edits were made in the utterance table or line number, regenerate the graph
    #       otherwise use the same graph for visualization changes
    if tokens_changed:
        G = generate_knowledge_graph(
            start=start_line, end=end_line, sentence_boost=False, with_interviewer=show_interviewer,
        )
        tokens_changed = False

    # first, remove edges that are below the degree offset value (like less than min degrees)
    edges_to_drop = [
        (e1, e2) for (e1, e2) in G.edges if G[e1][e2]["weight"] < min_co_occurrence
    ]
    G.remove_edges_from(edges_to_drop)

    # CALCULATE NODE METRICS

    # I add 1 to node size because if n=1 -> log2(1) = 0
    node_sizes = list(
        map(
            lambda x: 1 + np.log2(x) * size_multiplier,
            nx.get_node_attributes(G, "count").values(),
        )
    )
    node_degrees = dict(
        G.degree
    )  
    # because G.degree is a degreeview and doesn't have a values() method
    node_clustering = nx.clustering(G)
    d_centrality = nx.degree_centrality(G)
    b_centrality = nx.betweenness_centrality(G)

    # VISUALIZE

    # if showing a cumulative graph, just show the current line number
    #    otherwise, show the range of the line numbers
    plot_header = f"{end_line}" if start_line == 0 else f"[{start_line},{end_line}]"

    # create node label but don't show labels for nodes with only 1 count

    # find the most central node so that we can show labels of the nodes in its ego graph in the plot
    #      but hide the labels of the others for easier viewing
    most_central_node = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]

    ego_network = nx.ego_graph(G, n=most_central_node, radius=10)

    # show the node texts for really large nodes or the ones in the central node's plot
    node_texts = [
        nlp.vocab.strings[n]
        if G.nodes[n]["count"] > 5 or n in ego_network.nodes
        else " "
        for n in G.nodes
    ]

    # hover text for additional information for each node
    hover_texts = [
        f'<b>{G.nodes[node]["label"]}</b> <br> '
        f'𝑓: {G.nodes[node]["count"]} <br> '
        f"deg: {node_degrees[node]} <br>"
        f"clustering: {node_clustering[node]:.3f} <br>"
        f"degree centrality: {d_centrality[node]:.3f} <br>"
        f"betweenness centrality: {b_centrality[node]:.3f} <br>"
        for node in G.nodes
    ]

    # generate the selected layout for node and edge positions
    layout_seed = np.random.RandomState(42)

    pos = dict()

    if layout == "1":
        pos = nx.spring_layout(
            G, iterations=spring_iterations, seed=layout_seed, k=spring_k
        )

    if layout == "2":
        pos = nx.random_layout(G, seed=layout_seed)

    if layout == "3":
        pos = nx.shell_layout(G)

    if layout == "4":
        pos = nx.circular_layout(G)

    # create the plotly graph for the network
    edge_x = []
    edge_y = []

    light_edge_x = []
    light_edge_y = []

    if min_co_occurrence > min_dmc_co_occurrence:
        min_dmc_co_occurrence = min_co_occurrence

    for n1, n2 in G.edges():
        x0, y0 = pos[n1]
        x1, y1 = pos[n2]

        if G[n1][n2]["weight"] > min_dmc_co_occurrence:
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        else:
            light_edge_x.append(x0)
            light_edge_x.append(x1)
            light_edge_x.append(None)
            light_edge_y.append(y0)
            light_edge_y.append(y1)
            light_edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=2, color="#888"), mode="lines"
    )

    light_edge_trace = go.Scatter(
        x=light_edge_x,
        y=light_edge_y,
        line=dict(width=1, color="#BBB", dash="dot"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = [pos[n][0] for n in pos]
    node_y = [pos[n][1] for n in pos]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hovertext=hover_texts,
        hoverinfo="text",
        text=node_texts,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale="Portland",
            reversescale=False,
            color=list(node_degrees.values()),
            size=node_sizes,
            line_width=1,
            colorbar=dict(title=dict(text="degree")),
        ),
    )

    plot_title = "Cumulative" if start_line == 0 else "DMC"

    fig_graph = go.Figure(
        data=[light_edge_trace, edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"{case_name} | {plot_title} View @ at {plot_header}",
                x=0.5,
                xanchor="center",
            ),
            font=dict(size=16),
            hovermode="closest",
            height=600,
            margin=dict(l=0, r=0, t=40, b=40),
            showlegend=False,
            uirevision="none"
        ),
    )

    fig_graph.update_xaxes(showticklabels=False)
    fig_graph.update_yaxes(showticklabels=False)

    graph_network = dcc.Graph(figure=fig_graph, config={"displayModeBar": True})

    # metrics plots

    # graph_metrics = "This section is temporarily disabled!"

    node_labels = dict([(token, nlp.vocab.strings[token]) for token in G.nodes])
    node_degrees = [G.degree[token] for token in G.nodes]
    node_clustering = nx.clustering(G)
    ave_clustering = nx.average_clustering(G) if len(node_clustering) > 0 else 0

    
    # first create a sorted list of degrees for plotting as a scatter plot and histogram
    connected_nodes = dict(
        [(nlp.vocab.strings[token], G.degree[token])
         for token in G.nodes
         if G.degree[token] > 0]
    )
    
    # only attempt to plot if there are any tokens with degree higher than 0 
        # --> why wouldn't there be? if this is run earlier or tries to be run before transcript in
    if len(connected_nodes) > 0:
    
        ave_degree = (2 * G.number_of_edges()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    
        fig_metrics = make_subplots(rows=1, cols=2,
                                    subplot_titles=(
                                        f"μ<sub>degree</sub> = <b>{ave_degree:.3f}</b> | "
                                        f"n<sub>connected</sub> = {len(connected_nodes)} | "
                                        f"n<sub>total</sub> = {G.number_of_nodes()}",
                                        f"μ<sub>clustering</sub> = <b>{ave_clustering:.3f}</b>"
                                    ),
                                    )
        if ave_degree > 0:
            degree_labels, degree_degrees = zip(
                *list(sorted(connected_nodes.items(), key=lambda t: t[1], reverse=True)))
            fig_metrics.add_trace(
                go.Scatter(
                    y=degree_degrees,
                    x=degree_labels
                ),
                row=1, col=1
            )
    
            graph_metrics = dcc.Graph(figure=fig_metrics)
    
        # now let's get clustering coefficients for nodes if it's > 0
    
        clustered_nodes = dict(
            [(nlp.vocab.strings[token], node_clustering[token])
             for token in G.nodes
             if node_clustering[token] > 0]
        )
    
        # only display the plot if there are clusters
        if len(clustered_nodes) > 0:
            cluster_labels, cluster_coefficients = zip(
                *list(sorted(clustered_nodes.items(), key=lambda t: t[1], reverse=True)))
    
            fig_metrics.add_trace(
                go.Scatter(
                    y=cluster_coefficients,
                    x=cluster_labels
                ),
                row=1, col=2
            )
    
        fig_metrics.update_yaxes(row=1, col=1)
        fig_metrics.update_yaxes(row=1, col=2)
        fig_metrics.update_layout(showlegend=False,
                                  title=dict(
                                      text=f"density = {nx.density(G):.3f}",
                                      x=0.5, xanchor='center'),
                                  margin=dict(l=0, r=0, t=40, b=40)
                                  )

    return graph_network, graph_metrics


# ---- INTERFACE ----

# -- input section --

INPUT_FOLDER = "samples"

# creates Path object
input_folder_path = Path(INPUT_FOLDER)

file_list = ["__manual entry__"]

# checks if there is a path directory from creating the path object
if input_folder_path.is_dir():
    # gets all txt files
    text_files = [f.name for f in sorted(input_folder_path.glob("*.txt"))]
    if len(text_files) > 0:
        # adds each txt file to the file_list list
        file_list.extend(text_files)

input_file_dropdown = dbc.Select(
    file_list, id="input-file-dropdown", value="01_cj2.txt"
)

mode_name_input = dbc.Input(id="mode-name", value="", placeholder="Enter mode name ...")

raw_text_input = dbc.Textarea(
    placeholder="Copy and paste some text here.", value="", rows=10, id="raw-text"
)

parse_button = dbc.Button("Parse", id="parse-button", size="lg", n_clicks=0)

reset_button = dbc.Button(
    "Reset Mode",
    id="reset-button",
    color="danger",
    outline=True,
    class_name="ms-auto",
    n_clicks=0,
)

inclusion_options = dbc.Checklist(
    options=[
        {"label": "Display Timestamp", "value": 0},
        {"label": "Display Speaker", "value": 1},
        {"label": "Ignore Interviewer Utterances", "value": 2},
    ],
    value=[2],
    inline=True,
    class_name="mb-4",
    id="inclusion-options",
)

input_accordion = dbc.Accordion(
    [
        dbc.AccordionItem(
            [
                dbc.Row(
                    dbc.Col(
                        [dbc.Label("Input File:"), input_file_dropdown, html.P("")],
                        width=10,
                        lg=6,
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        [dbc.Label("Mode name:"), mode_name_input, html.P("")],
                        width=10,
                        lg=6,
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.Label("Transcript:"),
                            raw_text_input,
                        ]
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            #inclusion_options,
                            parse_button,
                        ],
                        class_name="mt-4",
                    ),
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                reset_button,
                            ],
                            class_name="d-flex align-items-end",
                        )
                    ]
                ),
                html.P(""),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    "", id="reset-message-div", className="ms-auto"
                                ),
                            ],
                            class_name="d-flex align-items-end",
                        )
                    ]
                ),
            ],
            title="Input",
            item_id="0",
        )
    ],
    active_item="0",
    id="input-accordion",
    className="my-4",
)

# -- utterances section --

utterances_accordion = dbc.Accordion(
    dbc.AccordionItem(
        [inclusion_options,
            html.Div(
                [
                    html.P(
                        "Processed text will be displayed here as a datatable.", # would we ever care about unprocessed text?
                        className="lead",
                    )
                ],
                id="utterances-div",
            )
        ],
        id="revise",
        title="Revise and Code",
    ),
    # active_item="1",  # collapsed by default
)

graph_button = dbc.Button(
    "Generate Graph", id="graph-button", size="lg", n_clicks=0, disabled=True
)

generate_div = html.Div([graph_button], className="border rounded p-4 my-4")

theoretical_codes_list = [
    "emergent",
    "centralized",
    "probabilistic",
    "deterministic",
    "feedback",
    "fitting",
    "levels",
    "slippage",
]

code_checkboxes_container = dbc.Container(
    "",
    fluid=True,
    class_name="d-flex align-content-start flex-wrap",
    id="code-checkboxes-container",
)

# -- graph view --

graph_type_row = dbc.Row(
    [
        dbc.Col(
            dbc.Checkbox(label="Theoretical Codes", id="include-codes", value=False),
            xs=12,
            md=6,
            xl=2,
            class_name="mt-3",
        ),
        dbc.Col(
            dbc.Checkbox(label="DMC Mode", id="dmc-mode", value=False),
            xs=12,
            md=6,
            xl=2,
            class_name="mt-3",
        ),
    ]
)

grap_layout_options_div = html.Div(
    [
        html.H4("Layout", className="my-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Span("Min Tokens per Line: ", className="me-4"),
                        dcc.Input(
                            id="min-tokens",
                            type="number",
                            min=1,
                            max=20,
                            step=1,
                            value=4,
                            style={"margin-top": "-6px"},
                        ),
                    ],
                    md=12,
                    xl=3,
                    class_name="d-flex mt-3",
                ),
                dbc.Col(
                    [
                        html.Span("Min Co-occurrence: ", className="me-4"),
                        dcc.Input(
                            id="min-co",
                            type="number",
                            min=1,
                            max=10,
                            step=1,
                            value=1,
                            style={"margin-top": "-6px"},
                        ),
                    ],
                    md=12,
                    xl=3,
                    class_name="d-flex mt-3",
                ),
                dbc.Col(
                    [
                        html.Span("Min DMC Co-occurrence: ", className="me-4"),
                        dcc.Input(
                            id="min-dmc-co",
                            type="number",
                            min=1,
                            max=10,
                            step=1,
                            value=2,
                            style={"margin-top": "-6px"},
                        ),
                    ],
                    md=12,
                    xl=3,
                    class_name="d-flex mt-3",
                ),
                dbc.Col(
                    [
                        html.Span("DMC Window: ", className="me-4"),
                        dcc.Input(
                            id="dmc-window",
                            type="number",
                            min=1,
                            max=11,
                            step=2,
                            value=3,
                            style={"margin-top": "-6px"},
                        ),
                    ],
                    md=12,
                    xl=3,
                    class_name="d-flex mt-3",
                ),
            ],
            class_name="my-4",
            justify="center",
        ),
        html.H4("Visualization", className="my-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Span("Layout: ", className="me-4"),
                        dbc.Select(
                            id="graph-layout",
                            options=[
                                {"label": "Spring", "value": "1"},
                                {"label": "Random", "value": "2"},
                                {"label": "Shell", "value": "3"},
                                {"label": "Circle", "value": "4"},
                            ],
                            value="1",
                            class_name="w-50",
                        ),
                    ],
                    md=12,
                    xl=3,
                    class_name="d-flex mt-3",
                ),
                dbc.Col(
                    [
                        html.Span("Node Size: ", className="me-4"),
                        dcc.Input(
                            id="node-size",
                            type="number",
                            min=1,
                            max=40,
                            step=1,
                            value=5,
                            style={"margin-top": "-6px"},
                            className="ms-4",
                        ),
                    ],
                    md=12,
                    xl=3,
                    class_name="d-flex mt-3",
                ),
                dbc.Col(
                    [
                        html.Span("Spring iterations: ", className="me-4"),
                        dcc.Input(
                            id="layout-iterations",
                            type="number",
                            min=0,
                            max=500,
                            step=1,
                            value=10,
                            style={"margin-top": "-6px"},
                        ),
                    ],
                    md=12,
                    xl=3,
                    class_name="d-flex mt-3",
                ),
                dbc.Col(
                    [
                        html.Span("Spring k: ", className="me-4"),
                        dcc.Input(
                            id="layout-k",
                            type="number",
                            min=0,
                            max=100,
                            step=0.05,
                            value=0.5,
                            style={"margin-top": "-6px"},
                        ),
                    ],
                    md=12,
                    xl=3,
                    class_name="d-flex mt-3",
                ),
            ],
            class_name="my-4",
            justify="center",
        ),
    ],
    className="my-4",
)

graph_view_options_div = html.Div(
    [
        html.H3("Knowledge Graph", className="mb-4"),
        html.P(" "),
        graph_type_row,
        html.P(" "),
        html.Div(
            "Knowledge graph will be displayed once you generate it.",
            id="graph-div",
            className="text-center",
        ),
        dcc.Slider(
            id="graph-slider",
            min=1,
            max=2,
            step=1,
            value=1,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
            className="my-4",
        ),
        grap_layout_options_div,
    ],
    className="border rounded p-4 my-4",
)

metrics_viewer_wrapper_div = html.Div(
    [
        html.H3("Metrics", className="mb-4"),
        html.P(" "),
        html.Div("Will be updated once the graph is generated.", id="metrics-div"),
    ],
    className="border rounded p-4 my-4",
)

# -- coding modal view --

coding_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Modify"), close_button=True),
        dbc.ModalBody(
            dbc.Row(
                [
                    dbc.Col("", id="token-buttons"),
                    dbc.Col(
                        [
                            dbc.Row(
                                dbc.Col(
                                    [
                                        html.H4("Frequency map"),
                                        html.Div(
                                            "Something must have gone wrong!",
                                            id="utterance-stats",
                                        ),
                                    ]
                                ),
                            ),
                            dbc.Row(
                                dbc.Col(
                                    [
                                        html.H4("Theoretical Codes"),
                                        html.P(
                                            "not yet functional",
                                            className="small text-italic",
                                        ),
                                        code_checkboxes_container,
                                    ]
                                ),
                                class_name="mt-4",
                            ),
                        ]
                    ),
                ]
            )
        ),
    ],
    id="coding-modal",
    scrollable=True,
    size="xl",
    is_open=False,
    centered=True,
)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H1(
                        ["mode-catcher ", html.Em("playground")],
                        className="text-center m-4",
                    ),
                    input_accordion                        
                ]
            )
        ),
        dbc.Row(dbc.Col(utterances_accordion)),
        dbc.Row(dbc.Col(generate_div)),
        dbc.Row(dbc.Col(graph_view_options_div)),
        dbc.Row(dbc.Col(metrics_viewer_wrapper_div)),
        coding_modal,
    ],
    fluid=True,
    class_name="p-4",
)


# ---- CALLBACKS ----
@app.callback(
    Output("raw-text", "value"),
    Output("mode-name", "value"),
    Input("input-file-dropdown", "value"),
)
def load_input_file(file_name: str):
    if file_name == "__manual entry__":
        return "", ""

    # gets path to file and removed .txt from the file's name
    file_path = Path(INPUT_FOLDER) / file_name
    mode_name = file_name.removesuffix(".txt")

    # checks file existence
    if not file_path.is_file():
        return "It doesn't seem like that file exists anymore.", mode_name

    # opens file and reads the file 
    # puts the text in one string instead of a list of lines
    with open(file_path, "r") as f:
        file_text = "".join([f"{line.strip()}\n" for line in f.readlines()])

    # checks if there is actually text (rather than empty file/string)
    if len(file_text) > 0:
        return file_text, mode_name

    return "File was there, but it had no text.", mode_name


@app.callback(
    Output("parse-button", "disabled"),
    Input("mode-name", "value"),
    Input("raw-text", "value"),
)
def activate_parse_button(name: str, text: str):
    if len(name.strip()) > 0 and len(text.strip()) > 0:
        return False

    return True


@app.callback(
    Output("reset-message-div", "children"),
    Input("reset-button", "n_clicks"),
    State("mode-name", "value"),
)
def reset_mode(nclicks, name):
    if ctx.triggered_id == "reset-button":
        # gets path of current model
        model_path = Path(f"./models/{str(name).strip()}/")

        # checks whether the path has directory
        if model_path.is_dir():
            # gets paths to specific files (stopwords and theoretical codes)
            stopwords_file = model_path / "stopwords.pickle"
            theoretical_codes_file = model_path / "theoretical_codes.pickle"

            # unlink deletes the pickled file (because it has been updated already?)
            if stopwords_file.is_file():
                stopwords_file.unlink()
            if theoretical_codes_file.is_file():
                theoretical_codes_file.unlink()

            for key in assigned_codes:
                assigned_codes[key] = [False] * len(theoretical_code_list)

            return "Existing mode files were cleared. Page refresh is recommended."

        else:
            return "No action taken because existing model couldn't be found."

@app.callback(
    Output("utterances-div", "children"),
    Output("input-accordion", "active_item"),
    Output("graph-button", "disabled"),
    Input("parse-button", "n_clicks"),
    State("inclusion-options", "value"),
    State("mode-name", "value"),
    State("raw-text", "value"),
    prevent_initial_call=True,
)
def utterance_table(parse_clicks, options, name, txt):
    global assigned_codes
    global nlp
    global stopped_words
    global unstopped_words
    global active_data

    if ctx.triggered_id == "parse-button":
        model_path = Path(f"./models/{str(name).strip()}/")
        default_stopwords_file = Path("./config") / "default_stopwords.pickle"

        # loading pickled files
        if model_path.is_dir():
            stopwords_file = model_path / "stopwords.pickle"
            theoretical_codes_file = model_path / "theoretical_codes.pickle"

            if stopwords_file.is_file():
                with open(stopwords_file, "rb") as swf:
                    stopped_words, unstopped_words = pickle.load(swf)
            else:
                with open(default_stopwords_file, "rb") as f:
                    stopped_words = pickle.load(f)

            if theoretical_codes_file.is_file():
                with open(theoretical_codes_file, "rb") as tcf:
                    saved_codes = pickle.load(tcf)

                assigned_codes = saved_codes
        else:
            with open(default_stopwords_file, "rb") as f:
                stopped_words = pickle.load(f)

        # reload the model because it only pulls default stopwords when loading
        nlp = spacy.load("en_core_web_sm", exclude=["ner", "senter"])

        # update stop_words of the small model
        #   I have to do it this y because spacy's to_disk method doesn't save stopwords
        for word in stopped_words:
            nlp.vocab[word].is_stop = True

        for word in unstopped_words:
            nlp.vocab[word].is_stop = False

        time = True
        speaker = True
        interviewer = True

        parsed_data = parse_raw_text(
            txt, timestamp=time, is_interviewer=interviewer
        )

        column_defs = [{'field': 'line', 'id': 'line', 'flex': 1, 'editable': True},
                       {'field': 'include?', 'id': 'include?', 'flex': 1, "boolean_value": True, "editable": True},
                       {'field': 'time', 'id': 'time', 'hide': 0 not in options},
                       {'field': 'speaker', 'id': 'speaker', 'hide': 1 not in options,
                        'filter': 'agSpeakerColumnFilter', 
                        'filterParams': {'comparator': {'function': 'speakerFilterComparator'}},
                        'isExternalFilterPresent': {'function': 2 in options},
                        'doesExternalFilterPass': 
                            {'function': "params.data.speaker != 'Interviewer'"}
                        },
                       {'field': 'utterance', 'id': 'utterance', 'flex': 5}]

        transcript_table = dag.AgGrid(
                    id = 'data-table',
                    rowData = parsed_data,
                    columnDefs = column_defs,
                    defaultColDef={
                        'resizable': True,
                        'cellStyle': {'wordBreak': 'normal'},
                        'cellRenderer': 'markdown',
                        'wrapText': True,
                        'autoHeight': True,
                        'filter': True},
                    dangerously_allow_code=True,
                    style={'height': 600})

        editor_section = [transcript_table]

        active_data = parsed_data

        return editor_section, "1", False
    else:
        message = [
            html.P(
                "Processed text will be displayed here as a datatable.",
                className="lead",
            )
        ]
        return message, "0", True

# needs to filter out interviewers as third otion
@app.callback(
    Output('data-table', 'columnState'),
    Output('data-table', 'dashGridOptions'),
    Input("inclusion-options", "value")
)
def helper(options):
    new_state = [{'colId': 'line'},
                    {'colId': 'include?'},
                    {'colId': 'time', 'hide': 0 not in options},
                    {'colId': 'speaker', 'hide': 1 not in options},
                    {'colId': 'utterance'}]
    new_filter = {'isExternalFilterPresent': {'function': 'false'}}
    if 2 in options:
        new_filter = {
            'isExternalFilterPresent': {'function': 'true'},
            'doesExternalFilterPass': 
                {'function': "params.data.speaker != 'Interviewer'"}
        }
    return new_state, new_filter


@app.callback(
    Output("token-buttons", "children"),
    Output("utterance-stats", "children"),
    Output("code-checkboxes-container", "children"),
    Output("coding-modal", "is_open"),
    Input("data-table", "cellClicked"),
    Input({"type": "toggle-token", "index": ALL, "stop": ALL}, "n_clicks"),
    Input({"type": "code-checkbox", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def coding_editor(cell, toggle_clicks, checked_codes):
    global active_data
    global tokens_changed

    if cell is not None:
        if len(toggle_clicks) > 0:
            if 1 in toggle_clicks:
                toggled_token = ctx.triggered_id["index"]
                was_stop = ctx.triggered_id["stop"]

                if was_stop:
                    nlp.vocab[toggled_token].is_stop = False
                    stopped_words.discard(toggled_token)
                    unstopped_words.add(toggled_token)
                else:
                    nlp.vocab[toggled_token].is_stop = True
                    stopped_words.add(toggled_token)
                    unstopped_words.discard(toggled_token)

                tokens_changed = True
        i = int(cell["rowId"])
        cell_text = str(active_data[i]["utterance"])
        token_buttons, token_treemap = process_utterance(cell_text)

        line_num = int(active_data[i]["line"] - 1)
        if len(checked_codes) > 0:
            if type(ctx.triggered_id) is not str:
                if ctx.triggered_id["type"] == "code-checkbox":
                    codes = generate_code_checkboxes(line_num, checked_codes)
                else:
                    codes = generate_code_checkboxes(line_num)
            else:
                codes = generate_code_checkboxes(line_num)
        else:
            codes = generate_code_checkboxes(line_num)

        return token_buttons, token_treemap, codes, True
    else:
        return "Something", "went", "wrong", False


@app.callback(
    Output("graph-div", "children"),
    Output("graph-slider", "max"),
    Output("graph-slider", "value"),
    Output("metrics-div", "children"),
    Output("min-dmc-co", "value"),
    Input("graph-button", "n_clicks"),
    Input("graph-slider", "value"),
    Input("include-codes", "value"),
    Input("dmc-mode", "value"),
    Input("dmc-window", "value"),
    Input("min-co", "value"),
    Input("min-dmc-co", "value"),
    Input("graph-layout", "value"),
    Input("layout-iterations", "value"),
    Input("layout-k", "value"),
    Input("node-size", "value"),
    Input("inclusion-options", "value"), # this has been added
    Input({"type": "toggle-token", "index": ALL, "stop": ALL}, "n_clicks"),
    Input("data-table", "cellValueChanged"),
    State("graph-button", "disabled"),
    State("mode-name", "value"),
    
    prevent_initial_call=True,
)
def knowledge_graph(
    n_clicks,
    line,
    code_pref,
    dmc,
    window,
    deg,
    dmc_deg,
    layout,
    iterations,
    k,
    multiplier,
    options,
    changed_stop,
    changed_include,
    disabled,
    name,
    
):
    global active_data
    global tokens_changed
    global has_generated

    if disabled:
        return (
            "You need to process some data.",
            1,
            1,
            "You need to process some data.",
            deg,
        )

    if ctx.triggered_id == "graph-button":
        has_generated = True

    if ctx.triggered_id == "graph-slider":
        tokens_changed = True

    if ctx.triggered_id == "min-co":
        tokens_changed = True

    if ctx.triggered_id == "min-dmc-co":
        tokens_changed = True

    if ctx.triggered_id == "inclusion-options":
        tokens_changed = True
    
    if not has_generated:
        return (
            "You need to process some data.",
            1,
            1,
            "You need to process some data.",
            deg,
        )

    # first, let's pickle the user generated model
    pickle_model(name)

    # make sure min co-occurrence is not larger than min dmc co-occurrence
    dmc_deg = deg + 1 if deg > dmc_deg - 1 else dmc_deg

    # display the latest utterance when generating a cumulative layout
    # if 2 in options: skip every other line
    # add a state checker to the above callback
    if not dmc and ctx.triggered_id == "graph-button":
        line = line if line != 1 else len(active_data)

    start = 0
    end = line

    end = end if end < len(active_data) else len(active_data)

    if dmc:
        r = int((window - 1) / 2)
        start = max(0, line - r)
        end = min(len(active_data), line + r)

        if window == 1:
            end = start + 1

    graph, stats = display_knowledge_graph(
        start_line=start,
        end_line=end,
        case_name=name,
        with_codes=code_pref,
        layout=layout,
        spring_iterations=iterations,
        spring_k=k,
        min_co_occurrence=deg,
        min_dmc_co_occurrence=dmc_deg,
        size_multiplier=multiplier,
        show_interviewer = 2 not in options,
    )

    return graph, len(active_data), line, stats, dmc_deg

@app.callback(
    Input("data-table", "cellValueChanged"),
)
def update_included_lines(changed):
    global tokens_changed

    if changed:
        i = int(changed[0]["rowId"])
        cell_incl = changed[0]['data']['include?']
        active_data[i]['include?'] = cell_incl
        tokens_changed = True

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    app.run_server(debug=True)