import asyncio
import math
import pickle
import re
import tomllib
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import spacy
from dash import Dash, ALL, ctx, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

# ---- INTERNAL MODULES
import nlp_functions as nlf
import filesystem_functions as fs

# ---- GLOBAL VARIABLES ----

nlp = spacy.blank("en")  # loading a blank model because we'll load the actual model later in the parse step

G = nx.Graph()

excluded_rows = set()
graph_button_clicked = False
tokens_excluded_from_lines = dict()
user_actions = list()

# ----- DASH APP CONFIGURATION -----

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = f"mode-catcher playground | dev build ({datetime.today()})"

# needed to be able to publish the script on Heroku
server = app.server


# ---- UTILITY FUNCTIONS ----

def flush_globals():

    global excluded_rows                # TODO: remove this and use the rowData which includes this info anyways
    global graph_button_clicked         # TODO: figure out why this is needed & how to get rid of it
    global nlp
    global tokens_excluded_from_lines
    global user_actions                 # TODO: convert to dcc.Store & then use the timestamp callback to update the table

    excluded_rows = None
    tokens_excluded_from_lines = None
    user_actions = None

    excluded_rows = set()
    graph_button_clicked = False
    nlp = spacy.blank("en")
    tokens_excluded_from_lines = dict()
    user_actions = list()



async def pickle_model(mode_name, full_row_data, spacy_model, is_sentencized, deductive_codes, stopped_tokens, unstopped_tokens):

    global tokens_excluded_from_lines
    global excluded_rows

    model_path = fs.get_model_path(mode_name, spacy_model, is_sentencized)

    # pickle the stop words changed by the user
    with open(model_path / fs.PARSED_DATA_FILENAME, "wb") as f:
        pickle.dump(full_row_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # pickle the stop words changed by the user
    with open(model_path / fs.STOPWORDS_FILENAME, "wb") as f:
        pickle.dump((stopped_tokens, unstopped_tokens), f, protocol=pickle.HIGHEST_PROTOCOL)

    # pickle the tokens that are excluded in individual lines by the user
    with open(model_path / fs.EXCLUDED_TOKENS_FILENAME, "wb") as f:
        pickle.dump(tokens_excluded_from_lines, f, protocol=pickle.HIGHEST_PROTOCOL)

    # pickle the rows that are completely excluded by the user
    with open(model_path / fs.EXCLUDED_ROWS_FILENAME, "wb") as f:
        pickle.dump(excluded_rows, f, protocol=pickle.HIGHEST_PROTOCOL)

    # pickle the user selected deductive codes
    with open(model_path / fs.ASSIGNED_CODES_FILENAME, "wb") as f:
        pickle.dump(deductive_codes, f, protocol=pickle.HIGHEST_PROTOCOL)

    # pickle the user actions log
    with open(model_path / fs.USER_ACTIONS_FILENAME, "wb") as f:
        pickle.dump(user_actions, f, protocol=pickle.HIGHEST_PROTOCOL)



def unpickle_defaults_and_model(mode_name, spacy_model, is_sentencized):

    flush_globals()

    global excluded_rows
    global tokens_excluded_from_lines
    global user_actions

    model_path = fs.get_model_path(mode_name, spacy_model, is_sentencized)

    # load deductive code definitions
    with open(fs.CONFIG_FOLDER / fs.DEDUCTIVE_LABEL_DEFINITIONS_FILENAME, "rb") as f:
        code_definitions = tomllib.load(f)

    # load the default stopwords list
    with open(fs.CONFIG_FOLDER / fs.DEFAULT_STOPWORDS_FILENAME, "rb") as f:
        stopped_tokens = pickle.load(f)

    # load previously parsed data (if it exists)
    parsed_data_file = model_path / fs.PARSED_DATA_FILENAME
    if parsed_data_file.is_file():
        with open(parsed_data_file, "rb") as f:
            full_row_data = pickle.load(f)
    else:
        full_row_data = list()

    # load the user-made changes to the stopwords (if they exist)
    stopwords_file = model_path / fs.STOPWORDS_FILENAME
    if stopwords_file.is_file():
        with open(stopwords_file, "rb") as f:
            stopped_tokens, unstopped_tokens = pickle.load(f)
    else:
        stopped_tokens = list()
        unstopped_tokens = list()

    # load the tokens that were excluded on specific lines by the user
    excluded_tokens_file = model_path / fs.EXCLUDED_TOKENS_FILENAME
    if excluded_tokens_file.is_file():
        with open(excluded_tokens_file, "rb") as f:
            tokens_excluded_from_lines = pickle.load(f)

    # load the lines that were completely excluded by the user
    excluded_rows_file = model_path / fs.EXCLUDED_ROWS_FILENAME
    if excluded_rows_file.is_file():
        with open(excluded_rows_file, "rb") as f:
            excluded_rows = pickle.load(f)

    # load the deductive codes selected by the user
    assigned_codes_file = model_path / fs.ASSIGNED_CODES_FILENAME
    if assigned_codes_file.is_file():
        with open(assigned_codes_file, "rb") as f:
            assigned_codes = pickle.load(f)
    else:
        assigned_codes = dict()

    # load the user actions log
    user_actions_log_file = model_path / fs.USER_ACTIONS_FILENAME
    if user_actions_log_file.is_file():
        with open(user_actions_log_file, "rb") as f:
            user_actions = pickle.load(f)

    return full_row_data, assigned_codes, code_definitions, stopped_tokens, unstopped_tokens


# ---- NLP FUNCTIONS ----

def parse_raw_text(txt: str,
                   timestamp=False,
                   is_interviewer=False,
                   in_sentences=True,
                   use_nlp_tags=False,
                   ):

    global excluded_rows
    global tokens_excluded_from_lines
    global nlp

    first_parse = True if len(tokens_excluded_from_lines) == 0 else False

    if excluded_rows is None:
        excluded_rows = list()

    data = list()

    # parse the text
    input_lines = [
        line.strip().replace("\n", "")
        for line in txt.splitlines()
        if len(line.strip()) > 0 and line.count(":") > 2
    ]

    # to parse the text line by line
    re_time_splitter = re.compile(r"(\[[0-9][0-9]:[0-9][0-9]:[0-9][0-9]])")

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
            'highlighted utterance': '',
            'in?': True
        }

        if timestamp:
            row['time'] = time

        if speaker:
            row['speaker'] = speaker

        doc = nlp(utterance.strip())

        if in_sentences:
            for s in doc.sents:

                excluded_in_row = tokens_excluded_from_lines.get(i, [])

                # if the user wants to filter out tokens based on NLP tags
                #    but only if this transcript is being loaded for the first time
                #    otherwise, don't overwrite user-made changes
                if use_nlp_tags and first_parse:
                    excluded_in_row.extend([ t.lemma_ for t in s if nlf.has_excluded_nlp_tag(t) and not t.is_stop ])

                # add the tokens excluded by the algorithm to the rest of exclusions
                if i in tokens_excluded_from_lines.keys():
                    tokens_excluded_from_lines[i].extend(excluded_in_row)
                else:
                    tokens_excluded_from_lines[i] = excluded_in_row

                # remove duplicate elements
                tokens_excluded_from_lines[i] = list(set(tokens_excluded_from_lines[i]))

                # create the row data to pass to the ag-grid
                sent_row = row.copy()
                i += 1
                sent_row['line'] = i
                sent_row['in?'] = False if i in excluded_rows else True

                utterance = s.text

                sent_row['utterance'] = utterance
                sent_row['tokens'] = s.as_doc().to_json()        # umit's addition onn 04/24 to avoid reparsing the sentences again and again
                data.append(sent_row)

        else:
        # here would I go through and make each token bold using markdown?
            i += 1
            row['line'] = i
            row['in?'] = False if i in excluded_rows else True
            row["utterance"] = utterance.strip()
            row['tokens'] = doc.to_json()             # umit's addition onn 04/24 to avoid reparsing the sentences again and again
            data.append(row)

    return data


def generate_code_checkboxes(line_num, assigned_codes, code_definitions):

    line = str(line_num) ## Umit's note: I noticed that the saved deductive codes loaded with str indexes (04/22/2025)

    if assigned_codes is None:
        assigned_codes = dict()

    ## Create an empty list of values if the user did not select any values for this line
    if line not in assigned_codes.keys():
        assigned_codes[line] = dict()

    ## Umit's note on 03/19/2025:
    ##  I know the following nested list comprehension is a bit hard to read
    ##  but it is kind of the most efficient way to write this code

    checkboxes_container = html.Div(
        [
            dbc.Row([

                # category title
                dbc.Col(html.Span(category, className="fw-semibold"), width=12),

                # create the checkboxes & the popover
                dbc.Col(
                    [
                        dbc.Checklist(
                            id={
                                "type": "code-checklist",
                                "index": f"{category}"
                            },
                            options=[{"label": code, "value": code} for code in code_definitions[category].keys()],
                            label_checked_class_name="text-success",
                            value = assigned_codes[line][category] if category in assigned_codes[line].keys() else "",
                            inline=True,
                        ),
                        dbc.Popover(
                            [
                                dbc.PopoverHeader(category.replace("_", " "), class_name="fw-semibold"),
                                dbc.PopoverBody(
                                    [
                                        html.Div(
                                            [
                                                html.H5(
                                                    dbc.Badge(
                                                        code.replace("_", " "),
                                                        color="white",
                                                        text_color="primary",
                                                        className="border p-2 mt-3 mb-0",
                                                    )
                                                ),
                                                html.P(
                                                    html.Small(
                                                        html.Code(code_definitions[category][code]['keywords'])
                                                    ), className="ms-2",
                                                ),
                                                html.P([
                                                        html.Span("Conceptual Example: ", className="fw-medium"),
                                                        html.Br(),
                                                        html.Em(code_definitions[category][code]['conceptual_example'])
                                                    ], className="ms-2",
                                                ),

                                                html.P([
                                                        html.Span("Verbatim Excerpt: ", className="fw-medium"),
                                                        html.Br(),
                                                        html.Em(f"\"{code_definitions[category][code]['verbatim_excerpt']}\"")
                                                    ], className="ms-2",
                                                ),
                                            ],
                                        ) for code in code_definitions[category].keys()
                                    ],
                                    className="mb-4"
                                )
                            ],
                            target={
                                "type": "code-checklist",
                                "index": f"{category}"
                            },
                            placement="left",
                            trigger="hover",
                            # delay = {"show": 100, "hide": 20}  # leaving here in case we need to activate a delay in the future
                        )
                    ], width=12
                ),
            ], class_name="my-3") for category in code_definitions.keys()
        ],
        id="code-checkboxes-container",
    )

    return checkboxes_container

# mapping use of certain "tokens" --> words?
def process_utterance(raw_text, row):

    global nlp
    global tokens_excluded_from_lines

    doc = nlp(raw_text.strip().lower())

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
                    color="light" if nlp.vocab[token.lemma_].is_stop else "danger" if token.lemma_ in tokens_excluded_from_lines.get(row, []) else "success",
                    class_name="m-1",
                    size="sm",
                )
            )
            if not nlp.vocab[token.lemma_].is_punct
            else html.Span(token.text, className="mx-1")
            for token in doc
        ]
    )

    return buttons_for_text #, token_treemap




# ---- UTTERANCE TABLE ----

# create a highlighted version of any given utterance using the html <mark> tag
def highlight_utterance(line):
    global nlp
    global tokens_excluded_from_lines

    row = line["line"] - 1
    doc = nlp(line["utterance"]) # TODO -> make this line get the json doc from the line and use from spacy.tokens import Doc to recreate the doc object + see if it'd even give us any performance boost (perhaps not here, but when reloading existing data)
    line["highlighted utterance"] = "".join(t.text_with_ws if nlp.vocab[t.lemma].is_stop
                                                              or t.lemma_ in tokens_excluded_from_lines.get(row, [])
                                                              or t.is_punct else f"<mark>{t.text}</mark>{t.whitespace_}"
                                                            for t in doc)
    return line

# generate the highlighted utterance column values for the entire dataset
def generate_highlighted_utterances(data):
    return list(map(lambda x: highlight_utterance(x), data))

# created this function to refactor table generation because it was used in multiple places
def generate_utterance_table(data, display_options, in_sents=False):

    return dag.AgGrid(
        id='data-table',
        rowData=data,
        columnDefs=[
            {'field': 'line', 'headerName': 'Sent' if in_sents else 'Line', 'editable': False, 'maxWidth': 90},
            {'field': 'time', 'hide': 0 not in display_options, 'maxWidth': 120},
            {'field': 'speaker', 'hide': 1 not in display_options, 'maxWidth': 140, 'wrapText': False,
             'filter': 'agSpeakerColumnFilter',
             'filterParams': {'comparator': {'function': 'speakerFilterComparator'}},
             'isExternalFilterPresent': {'function': 2 in display_options},
             'doesExternalFilterPass': {'function': "params.data.speaker != 'Interviewer'"}
             },
            {'field': 'utterance', 'hide': 3 in display_options, 'flex': 1},
            {'field': 'highlighted utterance', 'headerName': 'Highlighted Utterance', 'hide': 3 not in display_options, 'flex': 1},
            {'field': 'in?', "boolean_value": True, "editable": True, 'maxWidth': 80},
            {'field': 'tokens', "editable": False, 'hide': True},
        ],
        defaultColDef={
            'resizable': True,
            'cellStyle': {'wordBreak': 'normal'},
            'cellRenderer': 'markdown',
            'wrapText': True,
            'autoHeight': True,
            'filter': True,
        },
        dashGridOptions={"rowHeight": 40}, # so that the height of single line rows are not recalculated in each update to prevent some interface jitteriness
        dangerously_allow_code=True, # to enable markdown rendering with the <mark> html tag because commonmark doesn't include highlighting
        columnSize="sizeToFit", # Umit's note: for some reason, using responsiveSizeToFit blocks hiding columns when an inclusion option is checked off
        style={'height': 600}
    )

# ---- NETWORK ANALYSIS

def generate_token_graph_object(
        data,
        start,
        end,
        assigned_codes,
        use_similarity=True,
        similarity_cutoff=0.8,
        use_deductive_codes=False,
        with_interviewer=False
):

    global nlp
    global tokens_excluded_from_lines

    new_G = nx.Graph()

    # if showing a cumulative graph (start == 0), generate nodes for just until that point
    #    otherwise, generate nodes for the entire transcript

    # TODO: phasing out the active_data object altogether and using ag_grid's rowData property
    #       embedding the parsed doc object as a hidden column so that we don't have to reparse it over and over again
    #           filtering start to end instead of refiltering entire dataset over and over again
    #       pickling the rowData of the ag_grid instead of the active data object so that it loads even faster

    data_dict_list = [row for row in data if row['line'] >= start and row['line'] <= end]

    for line in data_dict_list:

        if (with_interviewer or (not with_interviewer and line["speaker"].lower() != "interviewer")) and line['in?']:

            row = line["line"] - 1

            raw_utterance = line["utterance"].strip().lower()

            ##  if the user wants to include deductive codes in the anlysis,
            ##      append them as tokens at the end of the utterance
            if use_deductive_codes:
                row_str = str(row)
                if row_str in assigned_codes.keys():
                    raw_utterance = f"{raw_utterance} {' ' .join([' '.join(v) for v in assigned_codes[row_str].values() if v != ''])}"

            doc_line = nlp(raw_utterance) # cleans

            # exclude the following tokens from the graph:
            #   - punctuations
            #   - stop words
            #   - tokens whose lemmas are stop words
            #   - tokens which are manually excluded at specific lines
            tokens = [t.lemma for t in doc_line if not t.is_punct
                                                    and not t.is_stop
                                                    and not nlp.vocab[t.lemma_].is_stop
                                                    and not t.lemma_ in tokens_excluded_from_lines.get(row, [])
                      ]

            # incorporate deductive codes into the graph

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

    # combine similar tokens if the "combine-by-similarity" option is chosen
    #   using the contracted_nodes function of networkx
    if use_similarity:

        # first, let's make sure the token is in the model's vocab
        #   or the similarity algorithm will yield random results
        # also exlude tokens that are like numbers because it combines tokens like 15 & 20 otherwise :)
        tokens_in_vocab = [t for t in new_G.nodes if not nlp.vocab[t].is_oov and not nlp.vocab[t].like_num]

        # # now lets compare node pairs to see if we should combine them
        for n1, n2 in combinations(tokens_in_vocab, 2):

                # first, make sure we're not checking tokens that were already combined
                #         in a previous iteration of this loop
                if n1 in new_G.nodes and n2 in new_G.nodes:

                    # then check if the similarity between the two tokens is above the cutoff value
                    if nlp.vocab[n1].similarity(nlp.vocab[n2]) > similarity_cutoff:

                        # add the frequency of the second node to the first node
                        new_G.nodes[n1]["count"] += new_G.nodes[n2]["count"]

                        # add the label of the second node to the first node
                        new_G.nodes[n1]["label"] += f" <sup>+{nlp.vocab.strings[n2]}</sup> "

                        # Umit's NOTE: I did not implement any code that adjusts the weights of the 1st node's edges
                        #       based on the weights of the 2nd node's edges yet (because time :)

                        # finally combine the two tokens, which keeps the properties of the 1st node
                        new_G = nx.contracted_nodes(new_G, n1, n2, self_loops=True, copy=True)

    return new_G


def draw_token_graph_plotly_object(
        data,
        assigned_codes,
        start_line=0,  # if > 0, range mode is activated
        end_line=1,
        mode_name="",
        sentencized=False,
        spacy_model="en_core_web_sm",
        with_codes=False,
        layout=1,
        spring_iterations=30,
        spring_k=0.2,
        min_co_occurrence=1,
        min_strong_co_occurrence=2,
        size_multiplier=2,
        show_interviewer=False,
        show_all_labels=True,
        show_weak_links=True,
        combine_by_similarity=True,
        min_similarity=0.8
):
    global nlp
    global G

    # first, let's generate the token graph
    G = generate_token_graph_object(
        data= data,
        start=start_line,
        end=end_line,
        use_similarity=combine_by_similarity,
        similarity_cutoff=min_similarity,
        use_deductive_codes=with_codes,
        assigned_codes=assigned_codes,
        with_interviewer=show_interviewer,
    )

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

    layout_title = "Spring"
    if layout == "1":
        pos = nx.spring_layout(
            G, iterations=spring_iterations, seed=layout_seed, k=spring_k
        )

    if layout == "2":
        layout_title = "Random"
        pos = nx.random_layout(G, seed=layout_seed)

    if layout == "3":
        layout_title = "Shell"
        pos = nx.shell_layout(G)

    if layout == "4":
        layout_title = "Circular"
        pos = nx.circular_layout(G)

    # create the plotly graph for the network
    edge_x = list()
    edge_y = list()

    light_edge_x = list()
    light_edge_y = list()

    if min_co_occurrence > min_strong_co_occurrence:
        min_strong_co_occurrence = min_co_occurrence

    for n1, n2 in G.edges():
        x0, y0 = pos[n1]
        x1, y1 = pos[n2]

        if G[n1][n2]["weight"] > min_strong_co_occurrence:
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
            showscale=False,
            colorscale="Portland",
            reversescale=False,
            color=list(node_degrees.values()),
            size=node_sizes,
            line_width=1,
            # colorbar=dict(title=dict(text="degree")),
        ),
        customdata=list(G.nodes),
    )


    ## Create the subtitle text & config options dictionary that applies to both graphs

    timestamp = datetime.today().replace(microsecond=0)

    subtitle_user_choices = f"{'Sentences: ' if sentencized else 'Lines: '} [{start_line}, {end_line}] | weak={min_co_occurrence}; strong={min_strong_co_occurrence}+ | {layout_title if layout != '1' else f'Spring (k={spring_k}, {spring_iterations} iterations)'} | Model: {spacy_model.lstrip('en_core_web_')} | {f' Similarity < {min_similarity}' if combine_by_similarity else ''}{' | Includes Deductive Codes' if with_codes else ''}{' | Includes the Interviewer' if show_interviewer else ''} | {timestamp}"

    graph_config_options = dict(
        displaylogo=False,
        displayModeBar=True,
        doubleClick="reset+autosize",
        modeBarButtonsToRemove=["select2d", "lasso2d"],
        toImageButtonOptions=dict(
            filename=f"{mode_name}-{timestamp}",
            format="jpeg",
            scale=2,
        ),
    )

    fig_graph = go.Figure(
        data=[light_edge_trace, edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=mode_name,
                font=dict(size=20, weight="bold"),
                subtitle=dict(
                    text = subtitle_user_choices,
                    font = dict(size=12, color="gray")
                ),
                x=0.5,
                y=1,
                xanchor="center",
                yanchor="top",
            ),
            font=dict(size=16),
            hovermode="closest",
            height=600,
            margin=dict(l=0, r=0, t=80, b=40),
            showlegend=False,
            uirevision="none"
        ),
    )

    fig_graph.update_xaxes(showticklabels=False)
    fig_graph.update_yaxes(showticklabels=False)

    graph_network = dcc.Graph(
        id="graph-figure",
        figure=fig_graph,
        config=graph_config_options,
        animate=True,
        animation_options= dict(redraw=True, duration=100),
    )

    ## Graph Metrics PLOTS

    graph_metrics = html.P("No metrics to display yet because there are no connected tokens.",className="lead",)

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

        fig_metrics = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                f"μ<sub>deg</sub> = <b>{ave_degree:.3f}</b>",
                f"log-log",
                f"μ<sub>C</sub> = <b>{ave_clustering:.3f}</b>"
            ),
            column_widths=[0.5, 0.25, 0.25],
        )

        if ave_degree > 0:
            degree_labels, degree_degrees = zip(*list(sorted(connected_nodes.items(), key=lambda t: t[1], reverse=True)))

            # create the degree histogram
            fig_metrics.add_trace(
                go.Scatter(x=degree_labels, y=degree_degrees),
                row=1, col=1
            )

            # create a log-log plot of the degree distribution

            degree_histogram = nx.degree_histogram(G)

            fig_metrics.update_xaxes(title_text="$log_{10}(deg)$", row=1, col=2)
            fig_metrics.update_yaxes(title_text="$log_{10}(n)$", row=1, col=2)

            # calculate log-log points using the degree histogram
            #   but drop the 0 values

            loglog_points_x = list()
            loglog_points_y = list()

            for i in range(1, len(degree_histogram), 1):
                if degree_histogram[i] > 0:
                    log_x = math.log10(i)
                    log_y = math.log10(degree_histogram[i])

                    if log_y > 0:
                        loglog_points_x.append(log_x)
                        loglog_points_y.append(log_y)

            loglog_fit = np.polynomial.Polynomial.fit(loglog_points_x, loglog_points_y, 2)
            loglog_fitline_x = np.arange(0.01, max(loglog_points_x), 0.01)
            loglog_fitline_y = loglog_fit(loglog_fitline_x)

            fig_metrics.add_trace(
                go.Scatter(x=loglog_points_x, y=loglog_points_y, mode="markers"),
                row=1, col=2
            )

            fig_metrics.add_trace(
                go.Scatter(x=loglog_fitline_x, y=loglog_fitline_y, mode="lines"),
                row=1, col=2
            )

        # create the metric graph object

        graph_metrics = dcc.Graph(figure=fig_metrics, config=graph_config_options, mathjax=True)
    
        # get the clustering coefficients for nodes if it's > 0
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
                go.Scatter(x=cluster_labels, y=cluster_coefficients),
                row=1, col=3
            )

        fig_metrics.update_yaxes(row=1, col=1)
        fig_metrics.update_yaxes(row=1, col=2)
        fig_metrics.update_yaxes(row=1, col=3)

        fig_metrics.update_layout(
            margin=dict(l=0, r=0, t=80, b=40),
            showlegend=False,
            xaxis=dict(
                nticks=len(degree_labels),
                ticklabelstep=1 if len(degree_labels) < 50 else math.ceil(len(degree_labels) / 50), # shows all the tick labels if there are very few
                tickfont=dict(size=9),
            ),
            title=dict(
                text=f'metrics for {mode_name}',
                font=dict(size=18, weight="bold"),
                subtitle=dict(
                    text=f"n = {G.number_of_nodes()} (n<sub>d>0</sub> = {len(connected_nodes)}) | ρ = {nx.density(G):.3f} | {subtitle_user_choices}",
                    font=dict(size=10, color="gray")
                ),
                x=0.5,
                y=0.99,
                xanchor="center",
                yanchor="top"
            ),
        )

    return graph_network, graph_metrics




# ---- INTERFACE ----
# ---- INTERFACE ----
# ---- INTERFACE ----


# -- input section --
# creates Path object
INPUT_FOLDER = Path("samples")

file_list = ["__manual entry__"]

# checks if there is a path directory from creating the path object
if INPUT_FOLDER.is_dir():
    # gets all txt files
    text_files = [f.name for f in sorted(INPUT_FOLDER.glob("*.txt"))]
    if len(text_files) > 0:
        # adds each txt file to the file_list list
        file_list.extend(text_files)

input_file_dropdown = dbc.Select(
    file_list, id="input-file-dropdown", value="_demo_cory1_abc.txt", persistence=True,
)

model_name_input = dbc.Input(id="mode-name", value="", placeholder="Enter model name ...")

raw_text_input = dbc.Textarea(
    placeholder="Copy and paste some text here.", value="", rows=10, id="raw-text"
)

parse_button = dbc.Button("Parse", id="parse-button", size="lg", class_name="mx-1")

load_cached_button = dbc.Button("Reload", id="load-cached-button", outline=True, size="lg", color="primary", disabled=True, class_name="mx-2")

split_into_sents_checkbox = dbc.Checkbox(label="Split into sentences", id="split-into-sentences", value=True, persistence=True)
apply_tags_checkbox = dbc.Checkbox(label="Infer irrelevant tokens", id="use-nlp-tags", value=True, persistence=True)


model_selection_dropdown = dbc.InputGroup([
    dbc.InputGroupText("Model"),
    dbc.Select(
        id="model-selection-dropdown",
        options=[
            {"label": "Small", "value": "en_core_web_sm"},
            {"label": "Medium", "value": "en_core_web_md"},
            {"label": "Large", "value": "en_core_web_lg"},
        ],
        persistence=True,
        value="en_core_web_lg"
    )
], class_name="mb-2"),

reset_button = dbc.Button(
    "Reset Model",
    id="reset-button",
    color="danger",
    outline=True,
    class_name="ms-auto",
    size="sm"
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
                        width=12,
                        lg=8,
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        dbc.InputGroup([
                            dbc.InputGroupText("Mode name"),
                            model_name_input
                        ]),
                        class_name="mb-4",
                        width=12,
                        lg=8,
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.Label("Transcript:"),
                            raw_text_input,
                        ]
                    ),
                    class_name="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(model_selection_dropdown, xl=4),
                        dbc.Col(split_into_sents_checkbox, xl=2),
                        dbc.Col(apply_tags_checkbox, xl=3),
                    ],
                    class_name="mt-4",
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            parse_button,
                            load_cached_button,
                            reset_button,
                        ],
                        class_name="d-flex align-items-end my-4",
                    ),
                    
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div("", id="reset-message-div", className="ms-auto"),
                            class_name="d-flex align-items-end",
                        )
                    ],
                ),
            ],
            title="Input",
            item_id="input",
        )
    ],
    active_item="input",
    id="input-accordion",
    className="my-4",
)

# -- utterances section --

parsing_spinner = dbc.Spinner(html.Div(id="parsing-spinner"), color="primary"),

empty_utterances_table_data = [{
    'line': '0',
    'time': '00:00:00',
    'speaker': 'N/A',
    'utterance': 'Processed text will be displayed in this table.',
    'in?': False
}]

inclusion_options = dbc.Checklist(
    options=[
        {"label": "Display Timestamp", "value": 0},
        {"label": "Display Speaker", "value": 1},
        {"label": "Ignore Interviewer Speech", "value": 2},
        {"label": "Highlight Included Tokens", "value": 3},
    ],
    value=[0, 1, 2],
    inline=True,
    class_name="mb-4",
    id="inclusion-options-checklist",
    persistence=True,
)

utterances_accordion = dbc.Accordion(
    dbc.AccordionItem(
        [
            html.Div(
                [
                    inclusion_options,
                    dbc.Button("Clear Filters", id="clear-table-filters-button", size="sm", class_name="my-2"),
                ],
                className="d-flex justify-content-between",
            ),
            html.Div(
                [
                    generate_utterance_table(empty_utterances_table_data, (0, 1, 2), False)
                ],
                id="utterances-div",
            )
        ],
        item_id="revise",
        title="Revise Tokens",
    ),
    id="utterances-accordion",
    active_item="nil",  # collapsed by default
)

generate_div = html.Div([
    dbc.Row(
        [
            dbc.Col(
                dbc.Checkbox(id="use-deductive-codes", label="Include deductive codes", value=False, persistence=True),
                width=3
            ),
            dbc.Col(
                dbc.Checkbox(id="combine-by-similarity", label="Combine similar tokens", value=True, persistence=True),
                width=2
            ),
            dbc.Col(
                dbc.InputGroup([
                    dbc.InputGroupText("Min similarity"),
                    dbc.Input(id="min-similarity", type="number", min=0, max=1, step=0.01, value=0.8, disabled=False, persistence=True),
                ]), width=3
            ),
        ],
        align="center",
        class_name="mb-4"
    ),
    dbc.Row(
        dbc.Col(
            dbc.Button("Generate Graph", id="graph-button", size="lg", n_clicks=0, disabled=True)
        )
    ),
    ],
    className="border rounded p-4 my-4"
)

# -- user changes log section

# based on generate_utterance_table function
def generate_log_table(data, display_options):

    return dag.AgGrid(
        id='log-data-table',
        rowData=data,
        columnDefs=[
            {'field': 'time', 'hide': 0 not in display_options, 'maxWidth': 300},
            {'field': 'line', 'editable': False, 'maxWidth': 90},
            {'field': 'change', 'hide': 3 in display_options, 'flex': 1},
        ],
        defaultColDef={
            'resizable': True,
            'cellStyle': {'wordBreak': 'normal'},
            'cellRenderer': 'markdown',
            'wrapText': True,
            'autoHeight': True,
            'filter': True,
        },
        dashGridOptions={"rowHeight": 40}, # so that the height of single line rows are not recalculated in each update to prevent some interface jitteriness
        dangerously_allow_code=True, # to enable markdown rendering with the <mark> html tag because commonmark doesn't include highlighting
        columnSize="sizeToFit", # Umit's note: for some reason, using responsiveSizeToFit blocks hiding columns when an inclusion option is checked off
        style={'height': 600}
    )

empty_log_table_data = [{
    'time': 'YYYY-MM-DD 00:00:00',
    'line': '0',
    'change': 'User token changes will be displayed in this table.',
}]
if len(user_actions) > 0:
    empty_log_table_data = user_actions

user_log_accordion = dbc.Accordion(
    dbc.AccordionItem(
        [
            html.Div(
                [
                    generate_log_table(empty_log_table_data, (0, 1, 2))
                ],
                id="log-div",
            )
        ],
        id="log",
        title="User Actions",
    ),
    start_collapsed=True
)

# TO DO: add the table itself to the app and change it each time a user logs a change

# -- graph view --

grap_layout_options_div = html.Div(
    [
        html.Br(),
        html.H5("Graph Construction", className="my-2"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Weak min co-occurrence"),
                        dbc.Input(id="min-co", type="number", min=1, max=10, step=1, value=1, persistence=True, debounce=500),
                    ]),
                    class_name="mt-2",
                    md=6, lg=4, xl=3,
                ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Strong min co-occurrence"),
                        dbc.Input(id="min-strong-co", type="number", min=1, max=10, step=1, value=2, persistence=True, debounce=500)
                    ]),
                    class_name="mt-2",
                    md=6, lg=4, xl=3,
                ),
            ],
            class_name="mt-4"
        ),
        html.Br(),
        html.H5("Visualization", className="my-2"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Node Size"),
                        dbc.Input(id="node-size", type="number", min=1, max=40, step=1, value=5, persistence=True, debounce=500),
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
                                persistence=True
                            ),
                        ]),
                    ],
                    lg=5,
                    xl=3,
                ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Spring iterations"),
                        dbc.Input(id="layout-iterations", type="number", min=0, max=500, step=1, value=10, persistence=True, debounce=500),
                    ]),
                    lg=4,
                    xl=3,
                ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Spring k"),
                        dbc.Input(id="layout-k", type="number", min=0, max=100, step=0.05, value=0.5, persistence=True, debounce=500),
                    ]),
                    lg=3,
                    xl=2,
                ),
            ],
            class_name="mt-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Checkbox(label="Display weak links", id="weak-links", value=True, persistence=True),
                    ],
                    lg=3,
                    xl=2,
                ),
                dbc.Col(
                    [
                        dbc.Checkbox(label="Display all node labels", id="all-labels", value=True, persistence=True),
                    ],
                    lg=4,
                    xl=3,
                ),
            ],
            class_name="mt-4",
        ),dbc.Row(
            [
                dbc.Col([
                    dbc.Button("Set Default Parameters", id="reset-parameters", color="primary", outline=True, size="sm"),
                ]),
            ],
            class_name="mt-4",
        ),
    ],
    className="my-4",
)


graph_view_options_div = html.Div(
    [
        html.H4(
            [
                "Token Graph",
                dbc.Spinner(html.Div(id="graphing-spinner"), color="primary", size="md"),
            ], className="mb-4"
        ),
        html.Div(
            "The token graph will be displayed once you generate it.",
            id="graph-div",
            className="text-center border p-4",
        ),
        dcc.RangeSlider(
            id="graph-slider",
            step=None,
            marks={0: 'N/A'},
            value=[0, 0],
            tooltip={"placement": "bottom", "always_visible": True},
            className="my-4",
        ),
        grap_layout_options_div,
    ],
    className="border rounded p-4",
)

metrics_viewer_wrapper_div = html.Div(
    [
        html.H3("", className="mb-4"),
        html.P(" "),
        html.Div("This view will be updated once the graph is generated.", className="lead", id="metrics-div"),
    ],
    className="border rounded p-4 my-4",
)

# -- coding modal view --

coding_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Revise"), close_button=True),
        dbc.ModalBody(
            dbc.Row([
                dbc.Col([
                    html.H5("Tokens", className="mb-4 pe-4"),
                    html.Div(id="token-buttons")
                ]),
                dbc.Col(
                    [
                        html.H5("Deductive Codes"),
                        dbc.Container(id="code-checkboxes-container")
                    ]
                ),
            ])

        ),
        dbc.ModalFooter([
            html.H5("Color key: "),

            dbc.Button("stop word", id="stopword-key-button", color="light", class_name="m-1", size="sm"),
            dbc.Button("excluded only for this line", id="exclude-key-button", color="danger", class_name="m-1", size="sm",),
            dbc.Button("included as a node", id="include-key-button", color="success", class_name="m-1", size="sm",),

            # html.Small("* Deductive codes adapted from Jacobson (2001) and Chi (2005).", className="text-muted m-1"),

            dbc.Tooltip("Gray tokens are excluded from analysis for the entire transcript.", target="stopword-key-button", placement="right"),
            dbc.Tooltip("Red tokens are excluded from analysis only for this line but may be included in the other lines.", target="exclude-key-button", placement="right"),
            dbc.Tooltip("Yellow are included in the analysis.", target="include-key-button", placement="right"),

        ], class_name="d-flex justify-content-start"),
    ],
    id="coding-modal",
    scrollable=True,
    size="xl",
    is_open=False,
    centered=True,
)


## TODO: Remaining globals to convert to dcc.Store objects

# excluded_rows = set()
# graph_button_clicked = False
# lemmas_excluded_from_lines = dict()
# user_actions = list()

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [   
                    html.H1(
                        ["mode-catcher ", html.Em("playground", className="text-muted font-weight-light")],
                        className="text-center m-4",
                    ),
                    input_accordion                        
                ]
            )
        ),
        dbc.Row(dbc.Col(parsing_spinner), class_name="mb-4"),
        dbc.Row(dbc.Col(utterances_accordion)),
        dbc.Row(dbc.Col(generate_div)),
        dbc.Row(dbc.Col(graph_view_options_div)),
        dbc.Row(dbc.Col(metrics_viewer_wrapper_div)),
        dbc.Row(dbc.Col(user_log_accordion)),
        coding_modal,
        dcc.Store(id="modal-row-id"),  # to keep track of the id of the row that is being revised in the modal view
        dcc.Store(id="assigned-deductive-codes", data=dict()),
        dcc.Store(id="deductive-code-definitions", data=dict()),
        dcc.Store(id="stopped-tokens", data=list()),
        dcc.Store(id="unstopped-tokens", data=list()),
    ],
    fluid=True,
    class_name="p-4",
)




# ---- CALLBACKS ----


@app.callback(
    Output("raw-text", "value"),
    Output("mode-name", "value"),
    Output("load-cached-button", "disabled"),
    Output("load-cached-button", "color"),
    Input("input-file-dropdown", "value"),
    State("split-into-sentences", "value"),
    State("model-selection-dropdown", "value")
)
def load_input_file_callback(file_name, is_sentencized, spacy_model):
    if file_name == "__manual entry__":
        return "", "", True, "secondary"

    # gets path to file and removed .txt from the file's name
    file_path = INPUT_FOLDER / file_name
    mode_name = file_name.removesuffix(".txt")

    # checks file existence
    if not file_path.is_file():
        return "It doesn't seem like that file exists anymore.", mode_name, True, "secondary"

    # opens file and reads the file 
    # puts the text in one string instead of a list of lines
    with open(file_path, "r") as f:
        file_text = "".join([f"{line.strip()}\n" for line in f.readlines()])

    # checks if there is actually text (rather than empty file/string)
    if len(file_text) > 0:

        model_folder = fs.get_model_path(mode_name, spacy_model, is_sentencized)

        # TODO: review the following section about cached parsed data (probably not needed, repetitive, or erroneus)

        cached_parsed_data_file_name = model_folder / "parsed_data.pickle"
        if cached_parsed_data_file_name.is_file():
            return file_text, mode_name, False, "primary"
        else:
            return file_text, mode_name, True, "secondary"

    return "File was there, but it had no text.", mode_name, True, "secondary"


@app.callback(
    Output("parse-button", "disabled"),
    Input("mode-name", "value"),
    Input("raw-text", "value"),
)
def enable_parse_button_callback(name: str, text: str):
    return False if len(name.strip()) > 0 and len(text.strip()) > 0 else True


@app.callback(
    Output("reset-message-div", "children"),
    Input("reset-button", "n_clicks"),
    State("mode-name", "value"),
    State("split-into-sentences", "value"),
    State("model-selection-dropdown", "value"),
    prevent_initial_call=True,
)
def reset_model_button_callback(n_reset_clicks, mode_name, is_sentencized, spacy_model):

    if ctx.triggered_id == "reset-button":

        # first, let's get rid of the existing user generated model files

        ## path of current model folder
        model_folder = fs.get_model_path(mode_name, spacy_model, is_sentencized)

        ## checks whether a model folder exists
        if model_folder.is_dir():

            # first, delete all the files in the folder (unlink == delete)
            for pickle_file in model_folder.iterdir():
                pickle_file.unlink()

            ## then, remove the folder itself
            model_folder.rmdir()

        flush_globals()

        # now let's trigger a page refresh

        return dbc.Alert(
            [
                html.I(className="bi bi-info-circle-fill me-2"),
                dbc.Badge(
                    f'{mode_name} / {model_folder.name}',
                    color="light",
                    text_color="danger",
                ),
                html.Span(" was purged! A page refresh is highly recommended."),
            ],
            color="danger"
        )

    else:
        return ""



@app.callback(
    Output("data-table", "rowData"),
    Output("utterances-accordion", "active_item"),
    Output("input-accordion", "active_item"),
    Output("graph-button", "disabled"),
    Output("deductive-code-definitions", "data"),
    Output("assigned-deductive-codes", "data", allow_duplicate=True),
    Output("stopped-tokens", "data", allow_duplicate=True),
    Output("unstopped-tokens", "data", allow_duplicate=True),
    Output("parsing-spinner", "children"),

    Input("parse-button", "n_clicks"),
    Input("load-cached-button", "n_clicks"),
    State("input-file-dropdown", "value"),
    State("mode-name", "value"),
    State("raw-text", "value"),
    State("split-into-sentences", "value"),
    State("model-selection-dropdown", "value"),
    State("use-nlp-tags", "value"),

    prevent_initial_call=True,
)
def parse_button_callback(
        n_parse_button_clicks,
        n_load_cached_button_clicks,
        filename,
        mode_name,
        raw_transcript_text,
        sentencized,
        spacy_model,
        use_nlp_tags,
):

    # first, reset all the globals
    #   to make sure that switching between transcripts doesn't mess things up
    flush_globals()

    global user_actions
    global excluded_rows
    global tokens_excluded_from_lines
    global nlp

    ## TODO: doing right now -> using row data instead of active_data + removing the excluded_rows global variable
    # excluded_rows = []

    cached_parsed_data, deductive_codes, code_definitions, stopped_tokens, unstopped_tokens = unpickle_defaults_and_model(mode_name, spacy_model, sentencized)

    # load the model selected by the user
    nlp = spacy.load(spacy_model, exclude=["ner"])

    # if resolve_corefs:
    #     nlp.add_pipe("fastcoref", config={  'device': 'cpu',
    #                                                 # 'model_architecture': 'LingMessCoref', # this model runs slower
    #                                                 # 'model_path': 'biu-nlp/lingmess-coref' # comment these two lines if you want the default faster model
    #                                              })


    # update stop_words of the small model
    #   I have to do it this y because spacy's to_disk method doesn't save stopwords
    for word in stopped_tokens:
        nlp.vocab[word].is_stop = True

    for word in unstopped_tokens:
        nlp.vocab[word].is_stop = False

    # tokens that are excluded from a specific line, but not the entire analysis
    time = True
    interviewer = True

    if ctx.triggered_id == "parse-button":
        # here in possible changes
        full_row_data = parse_raw_text(
            raw_transcript_text, timestamp=time,
            is_interviewer=interviewer,
            in_sentences = sentencized,
            use_nlp_tags = use_nlp_tags,
        )
    else:
        full_row_data = cached_parsed_data

    # save a backup of the input file (if it doesn't exist)
    # and save the input in the raw input texarea to the text file
    # so that the changes user makes in the raw input isn't lost.

    # TODO: Implement a new button that recovers the original text file

    # If the user manually typed a transcript, save it as a new txt file
    # otherwise, use the txt filename chosen on the dropdown
    if filename == "__manual entry__":
        input_file = INPUT_FOLDER / f'{mode_name}.txt'
    else:
        input_file = INPUT_FOLDER / filename

    backup_file = INPUT_FOLDER / f'{filename}backup'

    if not backup_file.is_file() and input_file.is_file():
        input_file.rename(backup_file)

    input_file.write_text(raw_transcript_text)

    # TODO -> refresh the file list if a new file was created and chose that file as the new input (requires updating this callback signature)
    # if filename == "__manual entry__":

    return generate_highlighted_utterances(full_row_data), "revise", "nil", False, code_definitions, deductive_codes, list(stopped_tokens), list(unstopped_tokens), ""



@app.callback(
    Output("clear-table-filters-button", "disabled"),
    Input("data-table", "filterModel"),
)
def activate_clear_table_filters_button_callback(existing_filters):

    # there is always
    keys = existing_filters.keys()
    n_filters = len(keys)
    if ('speaker' in keys and n_filters > 1) or ('speaker' not in keys and n_filters > 0):
        return False
    else:
        return True


@app.callback(
    Output('data-table', 'columnState'),
    Output('data-table', 'filterModel'),
    Input("inclusion-options-checklist", "value"),
    Input("clear-table-filters-button", "n_clicks"),
    State('data-table', 'filterModel'),
)
def apply_table_layout_filters_callback(table_display_options, n_reset_filters_clicks, existing_filters):

    new_state = [
        {'colId': 'line'},
        {'colId': 'time', 'hide': 0 not in table_display_options},
        {'colId': 'speaker', 'hide': 1 not in table_display_options},
        {'colId': 'utterance', 'hide': 3 in table_display_options, 'flex':1},
        {'colId': 'highlighted utterance', 'hide': 3 not in table_display_options, 'flex':1},
        {'colId': 'in?'},
    ]

    # if the "clear filters" button is clicked -> pass an empty dictionary
    #       otherwise, pass along the existing filters
    new_filters = existing_filters if ctx.triggered_id != "clear-table-filters-button" and existing_filters is not None else dict()

    # if the "ignore interviewer speech" option is selected
    if 2 in table_display_options:
        new_filters['speaker'] = {'filterType': 'text', 'type': 'notContains', 'filter': 'Interviewer'}

    return new_state, new_filters


@app.callback(
    Output("token-buttons", "children"),
    # Output("utterance-stats", "children"), # umit temporarily commented out this line on 02/24/2025 to deactivate the treemap visualization
    Output("code-checkboxes-container", "children"),
    Output("coding-modal", "is_open"),
    Output("modal-row-id", "data"),
    Output("stopped-tokens", "data", allow_duplicate=True),
    Output("unstopped-tokens", "data", allow_duplicate=True),
    Input("data-table", "cellClicked"),
    Input({"type": "toggle-token", "index": ALL, "stop": ALL}, "n_clicks"),
    State("data-table", "rowData"),
    State("assigned-deductive-codes", "data"),
    State("deductive-code-definitions", "data"),
    State("stopped-tokens", "data"),
    State("unstopped-tokens", "data"),
    prevent_initial_call=True,
)
def revise_tokens_view_callback(cell, toggle_clicks, row_data, assigned_codes, code_definitions, stopped_tokens, unstopped_tokens):

    global tokens_excluded_from_lines
    global user_actions

    # Umit's note on 04/22/2025: dcc.Store component does not support sets, so we have to convert them from lists, and vice versa
    stopped_tokens_set = set(stopped_tokens)
    unstopped_tokens_set = set(unstopped_tokens)

    # create global table
    if cell is not None:

        row = int(cell["rowId"])

        if type(ctx.triggered_id) != str:

            toggled_token = ctx.triggered_id["index"]
            was_stop = ctx.triggered_id["stop"]

            # to log the time this token was toggled
            curr_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            if was_stop:

                nlp.vocab[toggled_token].is_stop = False
                stopped_tokens_set.discard(toggled_token)
                unstopped_tokens_set.add(toggled_token)
                user_actions.append({'time': curr_time, 'line': row, 'change': f'\"{toggled_token}\" was toggled ON.\n'})

            else:

                # if a token was not a stop word, first check if it is in the excluded tokens list

                if toggled_token in tokens_excluded_from_lines.get(row, []):
                    # if it was an excluded token, turn it into a stop word
                    # and remove it from the exluded tokens list

                    nlp.vocab[toggled_token].is_stop = True
                    stopped_tokens_set.add(toggled_token)
                    unstopped_tokens_set.discard(toggled_token)
                    tokens_excluded_from_lines[row].remove(toggled_token)
                    user_actions.append({'time': curr_time, 'line': row, 'change': f'\"{toggled_token}\" was toggled OFF.\n'})

                else:

                    # if it was not in excluded tokens list, turn it into an excluded token

                    if len(tokens_excluded_from_lines.get(row, [])) == 0:
                        tokens_excluded_from_lines[row] = [toggled_token]
                    else:
                        tokens_excluded_from_lines[row].append(toggled_token)

                    user_actions.append({'time': curr_time, 'line': row, 'change': f'\"{toggled_token}\" was excluded from the line.'})
                    # change_log.append(html.P(f'At time {curr_time}: \"{toggled_token}\" was excluded from line {row + 1}.'))

        token_buttons = process_utterance(row_data[row]["utterance"], row=row)

        code_checkboxes = generate_code_checkboxes(row, assigned_codes, code_definitions)

        return token_buttons, code_checkboxes, True, row, list(stopped_tokens_set), list(unstopped_tokens_set)
    else:
        return "Something", "went wrong", False, -1, stopped_tokens, unstopped_tokens


@app.callback(
    Output("graph-div", "children"),
    Output("graph-slider", "marks"),
    Output("graph-slider", "value"),
    Output("metrics-div", "children"),
    Output("min-strong-co", "value"),
    Output("log-data-table", "rowData"),
    Output("graphing-spinner", "children"),

    Input("graph-button", "n_clicks"),
    Input("graph-slider", "value"),
    Input("min-co", "value"),
    Input("min-strong-co", "value"),
    Input("all-labels", "value"),
    Input("weak-links", "value"),
    Input("graph-layout", "value"),
    Input("layout-iterations", "value"),
    Input("layout-k", "value"),
    Input("node-size", "value"),
    # Input({"type": "toggle-token", "index": ALL, "stop": ALL}, "n_clicks"),

    State("inclusion-options-checklist", "value"),
    State("use-deductive-codes", "value"),
    State("assigned-deductive-codes", "data"),
    State("graph-button", "disabled"),
    State("mode-name", "value"),
    State("combine-by-similarity", "value"),
    State("min-similarity", "value"),
    State('data-table', 'virtualRowData'),
    State('data-table', 'rowData'),
    State("split-into-sentences", "value"),
    State("model-selection-dropdown", "value"),
    State("stopped-tokens", "data"),
    State("unstopped-tokens", "data"),

    prevent_initial_call=True,
)
def generate_graph_button_callback(
        n_graph_button_clicks,
        selected_range,
        min_co_occurrence,
        min_strong_co_occurrence,
        display_all_labels,
        display_weak_links,
        graph_layout,
        spring_iterations,
        spring_k,
        node_size_multiplier,
        selected_inclusion_options,
        use_deductive_codes,
        assigned_codes,
        graph_button_disabled,
        mode_name,
        combine_by_similarity,
        min_similarity,
        displayed_row_data,
        full_row_data,
        is_sentencized,
        spacy_model,
        stopped_tokens,
        unstopped_tokens
):

    global graph_button_clicked
    global user_actions

    if ctx.triggered_id == "graph-button":
        graph_button_clicked = True

        # also save (pickle) the user's work if the user clicks the "Generate Knowledge Graph" button
        #   and run that function asynchronously so that it doesn't slow down the graphing process
        asyncio.run(pickle_model(mode_name, full_row_data, spacy_model, is_sentencized, assigned_codes, stopped_tokens, unstopped_tokens))

    # prevents runtime errors if the user manually removed the values in these input ones to enter a new one
    if min_co_occurrence is None: min_co_occurrence = 1
    if min_strong_co_occurrence is None: min_strong_co_occurrence = 2

    # make sure min co-occurrence is not larger than min strong co-occurrence
    min_strong_co_occurrence = min_co_occurrence + 1 if min_co_occurrence > min_strong_co_occurrence - 1 else min_strong_co_occurrence

    # make the slider's tickers match the data at hand (has to be a dict)
    #   dictionary format is {line_num: 'label'}
    #   I left the labels empty so that the tooltip is the active label
    #   Otherwise, all numbers get jumbled up
    # I use sorted to make sure that the user sorting the table does not mess up the graph
    # I also make sure not to include the lines that were turned off by the user
    list_of_marks = sorted([l['line'] for l in displayed_row_data if l['in?']])
    slider_marks = {r: '' for r in list_of_marks}

    # determine the start and end of the range that the user picked
    start = selected_range[0]
    end = selected_range[1]
    if ctx.triggered_id == "graph-button":
        last_line = list(slider_marks.keys())[-1]
        end = selected_range[1] if selected_range[1] != 0 and selected_range[1] <= last_line else last_line

    end = end if end < displayed_row_data[-1]['line'] else displayed_row_data[-1]['line']

    selected_range = list([start, end])

    if spring_k == 0: spring_k = 0.05 # otherwise, networkx throws a `division by zero` error :)

    graph, stats = draw_token_graph_plotly_object(
        data=displayed_row_data,
        start_line=start,
        end_line=end,
        mode_name=mode_name,
        sentencized=is_sentencized,
        spacy_model=spacy_model,
        with_codes=use_deductive_codes,
        assigned_codes=assigned_codes,
        layout=graph_layout,
        spring_iterations=spring_iterations,
        spring_k=spring_k,
        min_co_occurrence=min_co_occurrence,
        min_strong_co_occurrence=min_strong_co_occurrence,
        size_multiplier=node_size_multiplier,
        show_interviewer =2 not in selected_inclusion_options,
        show_all_labels=display_all_labels,
        show_weak_links=display_weak_links,
        combine_by_similarity=combine_by_similarity,
        min_similarity=min_similarity,
    )

    return graph, slider_marks, selected_range, stats, min_strong_co_occurrence, user_actions, ""



# TODO: updating the datatable's highlighted tokens column once the user exits the revise modal

# Callback that happens after the revise modal view is closed
#   So that we an save the user selected deductive codes
@app.callback(
    Output("assigned-deductive-codes", "data", allow_duplicate=True),
    Input("coding-modal", "is_open"),
    State({"type": "code-checklist", "index": ALL}, "value"),
    State("modal-row-id", "data"),
    State("deductive-code-definitions", "data"),
    State("assigned-deductive-codes", "data"),
    prevent_initial_call=True,
)
def revise_modal_closed_callback(
        is_open,
        new_code_checkbox_values,
        row_id,
        deductive_code_definitions,
        previously_assigned_deductive_codes
):
    if is_open:
        # nothing to do if the modal was opened
        raise PreventUpdate
    else:

        # update the assigned deductive codes list once the modal is closed

        row_str = str(row_id)  # TODO: figure out why we have to convert this index to string :)
        updated_codes = previously_assigned_deductive_codes
        new_values = ctx.states_list[0]

        if row_str not in updated_codes.keys():
            updated_codes[row_str] = dict()

        for cat in new_values:
            cat_name = cat["id"]["index"]
            cat_vals = cat["value"]
            updated_codes[row_str][cat_name] = cat_vals

        return updated_codes


@app.callback(
    Output("log-data-table", "rowData", allow_duplicate=True),
    Input("data-table", "cellValueChanged"),
    prevent_initial_call=True,
)
def update_included_lines_callback(changed):

    global user_actions # TODO -> Pickle the row_data of the user actions table instead of this global variable

    if changed:
        i = int(changed[0]["rowId"])
        cell_incl = changed[0]['data']['in?']
        curr_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if cell_incl:
            excluded_rows.discard(i)
            text = f'The line is turned ON.'
        else:
            excluded_rows.add(i)
            text = f'The line is turned OFF.'
        user_actions.append({'time': curr_time, 'line': i + 1, 'change': text})
    return user_actions


@app.callback(
    Output("min-similarity", "disabled"),
    Input("combine-by-similarity", "value")
)
def toggle_min_similarity_input_callback(combine_by_similarity):
    return not combine_by_similarity


@app.callback(
    Output("min-co", "value", allow_duplicate=True),
    Output("min-strong-co", "value", allow_duplicate=True),
    Output("node-size", "value"),
    Output("graph-layout", "value"),
    Output("layout-iterations", "value"),
    Output("layout-k", "value"),
    Output("weak-links", "value"),
    Output("all-labels", "value"),
    Input("reset-parameters", "n_clicks"),
    prevent_initial_call=True,
)
def reset_parameters_callback(n_revert_button_clicks):
    return 1, 2, 5, "1", 10, 0.5, True, True

## Umit: I commented out the following callback on 04/17/2025 to not cause any troubles
##      but I'll work on implementing this `click-to-remove-node-from-graph` feature in May
##      but I'll work on implementing this `click-to-remove-node-from-graph` feature in May

# @app.callback(
#     Input("graph-figure", "clickData")
# )
# def test_graphobject_callback(click_data):
#
#     global nlp
#
#     if "points" in click_data.keys():
#         print(click_data["points"][0]["customdata"])

# --- RUN THE APP ---

if __name__ == "__main__":
    app.run(debug=True)