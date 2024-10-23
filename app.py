import pickle
import re
import os
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
import dash_auth
from itertools import combinations

from markdown_it.rules_core import inline
from plotly.subplots import make_subplots
import dash_ag_grid as dag
from datetime import datetime, time

# --- HEROKU SIMPLE AUTH PASSWORD ---

heroku_access_pwd = os.environ.get("CCL_ACCESS_PWD")

# ---- PLATFORM ----

nlp = spacy.blank("en")  # loading a blank model because we'll load the actual model later in the parse step

G = nx.Graph()

tokens_changed: bool = False
stopped_words = set()
unstopped_words = set()
assigned_codes = dict()
excluded_tokens = dict() # TODO: Change this to a hidden table column
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

change_log = []

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

# needed to be able to publish the script on Heroku
server = app.server

# ---- NLP ----

def parse_raw_text(txt: str, timestamp=False, is_interviewer=False, in_sentences=True, excluded_rows = [], use_nlp_tags=False):

    global excluded_tokens
    global nlp
    global tokens_changed

    first_parse = True if len(excluded_tokens) == 0 else False

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

    i = 0 # to count the number of sentences if the transcript is auto sentencized
    for line in input_lines:

        _, time, speaker_speech = re_time_splitter.split(line)
        speaker, utterance = speaker_speech.strip().split(":", maxsplit=1)
        speaker = str(speaker).strip()

        # initialize the dictionary
        row = {
            'line': 0,
            'time': '',
            'speaker': '',
            'utterance': '',
            'in?': True
        }

        if timestamp:
            row['time'] = time

        if speaker:
            row['speaker'] = speaker

        row['utterance'] = ''

        doc = nlp(utterance.strip())

        if in_sentences:
            for s in doc.sents:

                excluded_in_row = excluded_tokens.get(i, [])

                # if the user wants to filter out tokens based on NLP tags
                #    but only if this transcript is being loaded for the first time
                #    otherwise, don't overwrite user-made changes
                if use_nlp_tags and first_parse:
                    excluded_in_row.extend([ t.lemma_ for t in s if has_excluded_nlp_tag(t) and not t.is_stop ])

                # add the tokens excluded by the algorithm to the rest of exclusions
                if i in excluded_tokens.keys():
                    excluded_tokens[i].extend(excluded_in_row)
                else:
                    excluded_tokens[i] = excluded_in_row

                # remove duplicate elements
                excluded_tokens[i] = list(set(excluded_tokens[i]))

                # create the row data to pass to the ag-grid
                sent_row = row.copy()
                i += 1
                sent_row['line'] = i
                sent_row['in?'] = False if i in excluded_rows else True
                sent_row['utterance'] = s.text

                # create a highlighted version of the tokens
                # todo note: currently, this doesn't udpate after the user revises a line's tokens.
                sent_row['highlighted utterance'] = "".join(t.text_with_ws if nlp.vocab[t.lemma].is_stop or t.lemma_ in excluded_in_row or t.is_punct else f"<mark>{t.text}</mark>{t.whitespace_}" for t in s)

                data.append(sent_row)
        else:
        # here would I go through and make each token bold using markdown?
            i += 1
            row['line'] = i
            row["utterance"] = utterance.strip()
            row['in?'] = False if i in excluded_rows else True
            data.append(row)

        # if i not in assigned_codes.keys():
        #     assigned_codes[i] = [False] * len(theoretical_code_list) # initializing the assigned_codes dictionary

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
                        disabled=True,
                        # value=assigned_codes[line_num][code[0]],
                        id={"type": "code-checkbox", "index": code[1]}
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

# editable tag applications
def has_excluded_nlp_tag(token):

    # Parts of speech tags that should be automatically excluded
    # UH (3252815442139690129) == Interjection
    # IN (1292078113972184607) == Preposition

    # Dependency tags that should be automatically excluded
    # intj (421) == interjection
    # prep (443) == preposition
    # mark (423) == marker

    # TODO -> Use combinations of tag and dep to isolate tokens that are not meaningful
    #   for example an interjection such as "no" may be userful in some interviews, but not in all
    # TODO -> Design the final list of tag + dep combinations that should be excluded
    #      if the list of things to exclude is too large, we can instead focus on what to include

    return token.tag == 3252815442139690129 or token.tag == 1292078113972184607 or token.dep == 421 or token.dep == 423


# mapping use of certain "tokens" --> words?
def process_utterance(raw_text, row):

    global nlp
    global excluded_tokens

    doc = nlp(raw_text.strip().lower())

    all_tokens = [
        token.lemma_
        for token in doc
        if not nlp.vocab[token.lemma].is_stop
           and not token.is_punct
           and token not in excluded_tokens.get(row, [])
    ]

    buttons_for_text = html.Div(
        [
            html.Span(
                dbc.Button(
                    token.text,
                    id={
                        "type": "toggle-token",
                        "index": token.lemma_,
                        "stop": True if nlp.vocab[token.lemma_].is_stop else False
                    },
                    n_clicks=0,
                    color="light" if nlp.vocab[token.lemma_].is_stop else "danger" if token.lemma_ in excluded_tokens.get(row, []) else "warning",
                    class_name="m-1",
                    size="sm",
                )
            )
            if not nlp.vocab[token.lemma_].is_punct
            else html.Span(token.text, className="mx-1")
            for token in doc
        ]
    )

    token_counts = Counter(all_tokens)

    data_dict = {
        "token": list(token_counts.keys()),
        "count": list(token_counts.values()),
    }

    df = pd.DataFrame.from_dict(data_dict)

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

def pickle_model(mode_name, active_rows):
    global nlp
    global excluded_tokens

    models_folder = Path("./models/")
    models_folder.mkdir(exist_ok=True)

    mode_folder = models_folder / mode_name
    mode_folder.mkdir(exist_ok=True)

    # pickle the stop words changed by the user
    stopwords_file = mode_folder / "stopwords.pickle"

    with open(stopwords_file, "wb") as swf:
        pickle.dump(
            (stopped_words, unstopped_words), swf, protocol=pickle.HIGHEST_PROTOCOL
        )

    # pickle the tokens that are excluded in individual lines by the user

    excluded_tokens_file = mode_folder / "excluded_tokens.pickle"
    with open(excluded_tokens_file, "wb") as etf:
        pickle.dump(
            excluded_tokens, etf, protocol=pickle.HIGHEST_PROTOCOL
        )

    # pickle the rows that are completely excluded by the user
    excluded_rows = [l['line'] for l in active_rows if not l['in?']]
    excluded_rows_file = mode_folder / "excluded_rows.pickle"
    with open(excluded_rows_file, "wb") as erf:
        pickle.dump(
            excluded_rows, erf, protocol=pickle.HIGHEST_PROTOCOL
        )

    # umit temporarily disabled the following line(s)
    # theoretical_codes_file = mode_folder / "theoretical_codes.pickle"
    # with open(theoretical_codes_file, "wb") as tcf:
    #     pickle.dump(assigned_codes, tcf, protocol=pickle.HIGHEST_PROTOCOL)


def generate_knowledge_graph(start, end, with_interviewer=False):
    global nlp
    global active_data
    global excluded_tokens

    new_G = nx.Graph()

    # if showing a cumulative graph (start == 0), generate nodes for just until that point
    #    otherwise, generate nodes for the entire transcript
    data_dict_list = active_data[0:end] if start == 0 else active_data

    for line in data_dict_list:
        if ((with_interviewer or (not with_interviewer and line["speaker"].lower() != "interviewer"))
            and line['in?']):
            doc_line = nlp(line["utterance"].strip().lower()) # cleans

            row = line["line"] - 1

            # exclude the following tokens from the graph:
            #   - punctuations
            #   - stop words
            #   - tokens whose lemmas are stop words
            #   - tokens which are manually excluded at specific lines
            tokens = [t.lemma for t in doc_line if not t.is_punct
                                                    and not t.is_stop
                                                    and not nlp.vocab[t.lemma_].is_stop
                                                    and not t.lemma_ in excluded_tokens.get(row, [])
                      ]

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
    show_interviewer=False,
    show_all_labels=True,
    show_weak_links=True
):
    global nlp
    global G
    global tokens_changed

    # UA > if any edits were made in the utterance table or line number, regenerate the graph
    #       otherwise use the same graph for visualization changes
    if tokens_changed:

        # now let's generate the knowledge graph
        G = generate_knowledge_graph(
            start=start_line, end=end_line, with_interviewer=show_interviewer,
        )
        tokens_changed = False

    # first, remove edges that are below the degree offset value (like less than min degrees)
    edges_to_drop = [
        (e1, e2) for (e1, e2) in G.edges if G[e1][e2]["weight"] < min_co_occurrence
    ]
    G.remove_edges_from(edges_to_drop)

    # CALCULATE NODE METRICS

    # prevents a runtime error if the user manually removed the value to enter a new one
    if size_multiplier is None:
        size_multiplier = 1

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

    most_central_node = None

    if G.number_of_nodes() > 0:
        most_central_node = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]

    if most_central_node is not None:
        ego_network = nx.ego_graph(G, n=most_central_node, radius=10)
    else:
        ego_network = nx.empty_graph()

    # show the node texts for really large nodes or the ones in the central node's plot
    node_texts = [
        nlp.vocab.strings[n]
        if G.nodes[n]["count"] > 5 or n in ego_network.nodes or show_all_labels
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

    # prevents the division by zero error if the user manually removed the value to enter a new one
    if spring_iterations is None:
        spring_iterations = 1

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

    if show_weak_links:
        light_edge_trace = go.Scatter(
            x=light_edge_x,
            y=light_edge_y,
            line=dict(width=1, color="#BBB", dash="dot"),
            hoverinfo="none",
            mode="lines",
        )
    else:
        light_edge_trace = go.Scatter()

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
    else:
        graph_metrics = html.P("No metrics to display yet because there are no connected tokens.",className="lead",)

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
    file_list, id="input-file-dropdown", value="_demo_cory1_abc.txt"
)

mode_name_input = dbc.Input(id="mode-name", value="", placeholder="Enter mode name ...")

raw_text_input = dbc.Textarea(
    placeholder="Copy and paste some text here.", value="", rows=10, id="raw-text"
)

parse_button = dbc.Button("Parse", id="parse-button", size="lg", n_clicks=0)

sentencize_checkbox = dbc.Checkbox(label="Split into sentences?", id="by-sent", value=True)
apply_tags_checkbox = dbc.Checkbox(label="Use NLP tags to infer irrelevant tokens", id="use-nlp-tags", value=True)
model_selection_dropdown = dbc.Select(
    id="model-selection-dropdown",
    options=[
        {"label": "Small", "value": "en_core_web_sm"},
        {"label": "Medium", "value": "en_core_web_md"},
        {"label": "Large", "value": "en_core_web_lg", "disabled": False if heroku_access_pwd is None else True},
    ],
    value="en_core_web_sm"
)

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
        {"label": "Ignore Interviewer Speech", "value": 2},
        {"label": "Highlight tokens", "value": 3},
    ],
    value=[0, 1, 2],
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
                        dbc.InputGroup([
                            dbc.InputGroupText("Transcript File"),
                            input_file_dropdown
                        ]),
                        class_name="mb-4",
                        width=10,
                        lg=6,
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        dbc.InputGroup([
                            dbc.InputGroupText("Mode name"),
                            mode_name_input
                        ]),
                        class_name="mb-4",
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
                dbc.Row([
                    dbc.Col(sentencize_checkbox, xl=2),
                    dbc.Col(apply_tags_checkbox, xl=3),
                    dbc.Col(width=2),
                    dbc.Col(
                        dbc.InputGroup([
                            dbc.InputGroupText("Model"),
                            model_selection_dropdown
                        ]),
                        xl=3,
                        align="end",
                    )
                ], class_name="mt-4", justify="between"
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            parse_button
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
    "collective behavior"
    "centralized",
    "pre-determined",
    "god-like control",
    "probabilistic",
    "stochastic",
    "uncertainty",
    "randomness",
    "deterministic",
    "predictable",
    "monocausal",
    "multicausal",
    "non-linear",
    "criticality",
    "feedback",
    "fitting",
    "levels",
    "mid-level",
    "slippage",
    "dynamic equilibrium"
]

code_checkboxes_container = dbc.Container(
    "",
    fluid=True,
    class_name="d-flex align-content-start flex-wrap",
    id="code-checkboxes-container",
)

# -- graph view --

grap_layout_options_div = html.Div(
    [
        html.Br(),
        html.H4("Graph Construction", className="my-4"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Checkbox(label="Include Deductive Codes", id="include-codes", value=False, disabled=True),
                    xs=12,
                    md=6,
                    xl=2,
                ),
                dbc.Col(
                    [
                        dbc.Checkbox(label="DMC Mode", id="dmc-mode", value=False, disabled=True)
                    ],
                    xs=12,
                    md=6,
                    xl=2,
                ),
            ], class_name="mt-3"
        ),
        dbc.Row([
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Weak min co-occurrence"),
                        dbc.Input(
                            id="min-co",
                            type="number",
                            min=1,
                            max=10,
                            step=1,
                            value=1
                        ),
                    ]),
                    lg=6,
                    xl=3,
                ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Strong min co-occurrence"),
                        dbc.Input(
                            id="min-dmc-co",
                            type="number",
                            min=1,
                            max=10,
                            step=1,
                            value=2,
                        )]
                    ),
                    lg=6,
                    xl=4,
                ),
            ],
            class_name="mt-4"
        ),
        html.Br(),
        html.H4("Visualization", className="my-4"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Node Size"),
                        dbc.Input(
                            id="node-size",
                            type="number",
                            min=1,
                            max=40,
                            step=1,
                            value=5,
                        ),
                    ]),
                    lg=3,
                    xl=2,
                ),
                dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("Layout"),
                            dbc.Select(
                                id="graph-layout",
                                options=[
                                    {"label": "Circle", "value": "4"},
                                    {"label": "Random", "value": "2"},
                                    {"label": "Shell", "value": "3"},
                                    {"label": "Spring", "value": "1"},
                                ],
                                value="1",
                            ),
                        ]),
                    ],
                    lg=5,
                    xl=3,
                ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Spring iterations"),
                        dbc.Input(
                            id="layout-iterations",
                            type="number",
                            min=0,
                            max=500,
                            step=1,
                            value=10,
                        ),
                    ]),
                    lg=4,
                    xl=3,
                ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Spring k"),
                        dbc.Input(
                            id="layout-k",
                            type="number",
                            min=0,
                            max=100,
                            step=0.05,
                            value=0.5,
                        ),
                    ]),
                    lg=3,
                    xl=2,
                ),
            ],
            class_name="mt-4",
        ),
        dbc.Row([
                dbc.Col(
                    [
                        dbc.Checkbox(label="Display weak links?", id="weak-links", value=True),
                    ],
                    lg=3,
                    xl=2,
                ),
                dbc.Col(
                    [
                        dbc.Checkbox(label="Show all node labels?", id="all-labels", value=True),
                    ],
                    lg=4,
                    xl=3,
                ),
        ], class_name="mt-4",
        ),
    ],
    className="my-4",
)


graph_view_options_div = html.Div(
    [
        html.H3("Token Graph", className="mb-4"),
        html.P(" "),
        html.P(" "),
        html.Div(
            "The token graph will be displayed once you generate it.",
            id="graph-div",
            className="text-center",
        ),
        dcc.Slider(
            id="graph-slider",
            step=None,
            marks={0: 'N/A'},
            value=0,
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
        html.Div("This view will be updated once the graph is generated.", className="lead", id="metrics-div"),
    ],
    className="border rounded p-4 my-4",
)

change_log_viewer_wrapper_div = html.Div(
    [
        html.H3("User Actions", className="mb-4"),
        html.P(" "),
        html.Div(html.P("This view will be updated when the user toggles tokens.", className="lead"), id="changes-div"),
    ],
    className="border rounded p-4 my-4",
)

# -- coding modal view --

coding_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Revise Tokens"), close_button=True),
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
                                        dbc.Badge("not implemented", text_color="danger", color="white", className="border small text-italic"),
                                        html.H4("Deductive Codes"),
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
        dbc.Row(dbc.Col(change_log_viewer_wrapper_div)),
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
            # umit temporarily disabled the following line(s)
            # theoretical_codes_file = model_path / "theoretical_codes.pickle"

            # unlink deletes the pickled file (because it has been updated already?)
            if stopwords_file.is_file():
                stopwords_file.unlink()
            
            # umit temporarily disabled the following line(s)
            # if theoretical_codes_file.is_file():
            #     theoretical_codes_file.unlink()

            # for key in assigned_codes:
            #     assigned_codes[key] = [False] * len(theoretical_code_list)

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
    State("by-sent", "value"),
    State("model-selection-dropdown", "value"),
    State("use-nlp-tags", "value"),
    prevent_initial_call=True,
)
def utterance_table(parse_clicks, options, name, txt, sentencize, model, use_nlp_tags):
    global active_data
    global assigned_codes
    global excluded_tokens
    global nlp
    global stopped_words
    global tokens_changed
    global unstopped_words


    # first, reset all the globals
    #   to make sure that switching between transcripts doesn't mess things up
    active_data = list()
    assigned_codes = dict()
    excluded_tokens = dict()
    stopped_words = set()
    tokens_changed = True
    unstopped_words = set()

    excluded_rows = []

    if ctx.triggered_id == "parse-button":
        model_path = Path(f"./models/{str(name).strip()}/")
        default_stopwords_file = Path("./config") / "default_stopwords.pickle"

        # loading pickled files
        if model_path.is_dir():

            # load the user-made changes to the stopwords
            stopwords_file = model_path / "stopwords.pickle"

            if stopwords_file.is_file():
                with open(stopwords_file, "rb") as swf:
                    stopped_words, unstopped_words = pickle.load(swf)
            else:
                with open(default_stopwords_file, "rb") as swf:
                    stopped_words = pickle.load(swf)

            # load the tokens that were excluded on specific lines by the user
            excluded_tokens_file = model_path / "excluded_tokens.pickle"

            if excluded_tokens_file.is_file():
                with open(excluded_tokens_file, "rb") as etf:
                    excluded_tokens = pickle.load(etf)
            else:
                excluded_tokens = dict()

            # load the lines that were completely excluded by the user

            excluded_rows_file = model_path / "excluded_rows.pickle"

            if excluded_rows_file.is_file():
                with open(excluded_rows_file, "rb") as erf:
                    excluded_rows = pickle.load(erf)

            # umit temporarily disabled the following line(s)
            # theoretical_codes_file = model_path / "theoretical_codes.pickle"
            # if theoretical_codes_file.is_file():
            #     with open(theoretical_codes_file, "rb") as tcf:
            #         saved_codes = pickle.load(tcf)

                # assigned_codes = saved_codes
        else:
            with open(default_stopwords_file, "rb") as f:
                stopped_words = pickle.load(f)

        # reload the model because it only pulls default stopwords if loaded from the beginning
        nlp = spacy.load(model, exclude=["ner"])

        # update stop_words of the small model
        #   I have to do it this y because spacy's to_disk method doesn't save stopwords
        for word in stopped_words:
            nlp.vocab[word].is_stop = True

        for word in unstopped_words:
            nlp.vocab[word].is_stop = False

        # tokens that are excluded from a specific line, but not the entire analysis

        time = True
        speaker = True
        interviewer = True
        highlight = True if 3 in options else False

        # here in possible changes
        parsed_data = parse_raw_text(
            txt, timestamp=time,
            is_interviewer=interviewer,
            in_sentences = sentencize,
            excluded_rows = excluded_rows,
            use_nlp_tags = use_nlp_tags
        )

        column_defs = [
            {'field': 'line', 'headerName': 'Sent' if sentencize else 'Line', 'editable': False, 'maxWidth': 90},
            {'field': 'time', 'hide': 0 not in options, 'maxWidth': 120},
            {'field': 'speaker', 'hide': 1 not in options, 'maxWidth': 140, 'wrapText': False,
                'filter': 'agSpeakerColumnFilter',
                'filterParams': {'comparator': {'function': 'speakerFilterComparator'}},
                'isExternalFilterPresent': {'function': 2 in options},
                'doesExternalFilterPass': {'function': "params.data.speaker != 'Interviewer'"}
            },
            {'field': 'utterance', 'hide': 3 in options, 'flex': 1},
            {'field': 'highlighted utterance', 'headerName': 'Utterance', 'hide': 3 not in options, 'flex': 1},
            {'field': 'in?', "boolean_value": True, "editable": True, 'maxWidth': 80},
        ]

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
                        'filter': True,
                        },
                    dangerously_allow_code=True,
                    columnSize="sizeToFit", # for some reason, using responsiveSizeToFit blocks hiding columns when an inclusion option is checked off
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

# needs to filter out interviewers as third option
@app.callback(
    Output('data-table', 'columnState'),
    Output('data-table', 'dashGridOptions'),
    Input("inclusion-options", "value"),
)
def apply_table_layout_filters(options):

    new_state = [
        {'colId': 'line'},
        {'colId': 'time', 'hide': 0 not in options},
        {'colId': 'speaker', 'hide': 1 not in options},
        {'colId': 'utterance', 'hide': 3 in options, 'flex':1},
        {'colId': 'highlighted utterance', 'hide': 3 not in options, 'flex':1},
        {'colId': 'in?'},
    ]

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
    # State("use-nlp-tags", "value"),
    prevent_initial_call=True,
)
def revise_tokens_view(cell, toggle_clicks, checked_codes):
    global active_data
    global tokens_changed
    global excluded_tokens
    global change_log

    if cell is not None:

        row = int(cell["rowId"])

        if len(toggle_clicks) > 0:

            if 1 in toggle_clicks:
                toggled_token = ctx.triggered_id["index"]
                was_stop = ctx.triggered_id["stop"]
                # toggled token is here
                curr_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if was_stop:
                    nlp.vocab[toggled_token].is_stop = False
                    stopped_words.discard(toggled_token)
                    unstopped_words.add(toggled_token)
                    change_log.append(html.P(f'At time {curr_time}: \"{toggled_token}\" was toggled ON.\n'))
                else:
                    # if a token was not a stop word, first check if it is in the excluded tokens list

                    if toggled_token in excluded_tokens.get(row, []):
                        # if it was an excluded token, turn it into a stop word
                        # and remove it from the exluded tokens list

                        nlp.vocab[toggled_token].is_stop = True
                        stopped_words.add(toggled_token)
                        unstopped_words.discard(toggled_token)
                        excluded_tokens[row].remove(toggled_token)

                        change_log.append(html.P(f'At time {curr_time}: \"{toggled_token}\" was toggled OFF.'))

                    else:
                        # if it was not in excluded token list, turn it into an excluded token

                        if len(excluded_tokens[row]) == 0:
                            excluded_tokens[row] = [toggled_token]
                        else:
                            excluded_tokens[row].append(toggled_token)

                        change_log.append(html.P(f'At time {curr_time}: \"{toggled_token}\" was excluded from line {row + 1}.'))

                tokens_changed = True

        cell_text = str(active_data[row]["utterance"])
        token_buttons, token_treemap = process_utterance(cell_text, row=row)

        line_num = int(active_data[row]["line"] - 1)

        # umit temporarily disabled this code
        # if len(checked_codes) > 0:
        #     if type(ctx.triggered_id) is not str:
        #         if ctx.triggered_id["type"] == "code-checkbox":
        #             codes = generate_code_checkboxes(line_num, checked_codes)
        #         else:
        #             codes = generate_code_checkboxes(line_num)
        #     else:
        #         codes = generate_code_checkboxes(line_num)
        # else:
        #     codes = generate_code_checkboxes(line_num)

        codes = generate_code_checkboxes(line_num)

        return token_buttons, token_treemap, codes, True
    else:
        return "Something", "went", "wrong", False


@app.callback(
    Output("graph-div", "children"),
    Output("graph-slider", "marks"),
    Output("graph-slider", "value"),
    Output("metrics-div", "children"),
    Output("min-dmc-co", "value"),
    Output("changes-div", "children"),
    Input("graph-button", "n_clicks"),
    Input("graph-slider", "value"),
    Input("include-codes", "value"),
    Input("dmc-mode", "value"),
    Input("min-co", "value"),
    Input("min-dmc-co", "value"),
    Input("all-labels", "value"),
    Input("weak-links", "value"),
    Input("graph-layout", "value"),
    Input("layout-iterations", "value"),
    Input("layout-k", "value"),
    Input("node-size", "value"),
    Input("inclusion-options", "value"), # this has been added
    Input({"type": "toggle-token", "index": ALL, "stop": ALL}, "n_clicks"),
    # Input("data-table", "cellValueChanged"),
    State("graph-button", "disabled"),
    State("mode-name", "value"),
    State('data-table', 'virtualRowData'),
    prevent_initial_call=True,
)
def knowledge_graph(
    n_clicks,
    line,
    code_pref,
    dmc,
    deg,
    dmc_deg,
    all_labels,
    weak_links,
    layout,
    iterations,
    k,
    multiplier,
    options,
    changed_stop,
    # changed_include,
    disabled,
    name,
    active_row_data,
    prevent_initial_call=True,
):
    global active_data
    global tokens_changed
    global has_generated
    global change_log

    empty_return = ["You need to process some data.", {0: 'N/A'}, 0, "You need to process some data.", deg, "This view will be updated when the user toggles tokens."]

    if disabled:
        return empty_return

    if ctx.triggered_id == "graph-button":
        has_generated = True

        # also pickle the user's actions if the user clicks the "Generate Knowledge Graph" button
        pickle_model(name, active_row_data)

    if ctx.triggered_id == "graph-slider":
        tokens_changed = True

    if ctx.triggered_id == "min-co":
        tokens_changed = True

    if ctx.triggered_id == "min-dmc-co":
        tokens_changed = True

    if ctx.triggered_id == "inclusion-options":
        tokens_changed = True
    
    if not has_generated:
        return empty_return

    # prevents runtime errors if the user manually removed the values to enter a new one
    if deg is None: deg = 1
    if dmc_deg is None: dmc_deg = 2

    # make sure min co-occurrence is not larger than min dmc co-occurrence
    dmc_deg = deg + 1 if deg > dmc_deg - 1 else dmc_deg

    # make the slider's tickers match the data at hand (has to be a dict)
    #   dictionary format is {line_num: 'label'}
    #   I left the labels empty so that the tooltip is the active label
    #   Otherwise, all numbers get jumbled up
    # I use sorted to make sure that the user sorting the table does not mess up the graph
    # I also make sure not to include the lines that were turned off by the user
    list_of_marks = sorted([l['line'] for l in active_row_data if l['in?']])
    slider_marks = {r: '' for r in list_of_marks}

    # display the latest utterance when generating a cumulative layout
    # if 2 in options: skip every other line
    # add a state checker to the above callback
    if not dmc and ctx.triggered_id == "graph-button":
        last_line = list(slider_marks.keys())[-1]
        line = line if line != 0 and line <= last_line else last_line

    start = 0
    end = line

    end = end if end < len(active_data) else len(active_data)

    # if dmc:
    #     r = int((window - 1) / 2)
    #     start = max(0, line - r)
    #     end = min(len(active_data), line + r)
    #
    #     if window == 1:
    #         end = start + 1

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
        show_all_labels=all_labels,
        show_weak_links = weak_links,
    )

    return graph, slider_marks, line, stats, dmc_deg, change_log

@app.callback(
    Input("data-table", "cellValueChanged"),
)
def update_included_lines(changed):
    global tokens_changed

    if changed:
        i = int(changed[0]["rowId"])
        cell_incl = changed[0]['data']['in?']
        active_data[i]['in?'] = cell_incl
        tokens_changed = True

# --- HEROKU SIMPLE AUTH CHECK ---

heroku_access_pwd = os.environ.get("CCL_ACCESS_PWD")

if heroku_access_pwd:
    credentials_list = {"ccl" : heroku_access_pwd}
    auth = dash_auth.BasicAuth(app, credentials_list)

# --- RUN THE APP ---

if __name__ == "__main__":
    app.run(debug=True)