import pickle
import re
from collections import Counter
from pathlib import Path
import dash_bootstrap_components as dbc
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import spacy
from dash import Dash, ALL, ctx, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
from itertools import combinations
from plotly.subplots import make_subplots
import dash_ag_grid as dag
from datetime import datetime
import tomllib

# ---- INTERNAL MODULES
import nlp_functions as nlf

# ---- GLOBAL VARIABLES ----

nlp = spacy.blank("en")  # loading a blank model because we'll load the actual model later in the parse step

G = nx.Graph()

deductive_code_definitions = dict()  ## Keeps the info about the labels, not user selections
excluded_rows = set()
graph_button_clicked = False
graphed_tokens_changed = False  ## TODO: problematic global because once it's set to True, it remains True.
lemmas_excluded_from_lines = dict()
stopped_lemmas = set()
unstopped_lemmas = set()
user_actions = list()

# constants
MODELS_FOLDER = Path("./models/")
CONFIG_FOLDER = Path("./config/")


# ----- DASH APP CONFIGURATION -----

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

# needed to be able to publish the script on Heroku
server = app.server


# ---- UTILITY FUNCTIONS ----

def flush_globals():

    global deductive_code_definitions
    global excluded_rows
    global graph_button_clicked
    global graphed_tokens_changed
    global lemmas_excluded_from_lines
    global stopped_lemmas
    global unstopped_lemmas
    global user_actions

    deductive_code_definitions = None
    excluded_rows = None
    graph_button_clicked = None
    graphed_tokens_changed = None
    lemmas_excluded_from_lines = None
    stopped_lemmas = None
    unstopped_lemmas = None
    user_actions = None

    deductive_code_definitions = dict()  ## Keeps the info about the labels, not user selections
    excluded_rows = set()
    graph_button_clicked = False
    graphed_tokens_changed = False  ## TODO: problematic global because once it's set to True, it remains True.
    lemmas_excluded_from_lines = dict()
    stopped_lemmas = set()
    unstopped_lemmas = set()
    user_actions = list()



def get_model_path(mode_name, spacy_model, is_sentencized):
    ## create the models folder if it doesn't exist already
    ##     exists_ok = True -> don't overwrite if it already exists
    if not MODELS_FOLDER.is_dir():
        MODELS_FOLDER.mkdir(exist_ok=True)

    ## define the path of the mode (case, participant) for the selected transcript
    mode_path = MODELS_FOLDER / str(mode_name).strip()
    if not mode_path.exists():
        mode_path.mkdir(exist_ok=True)

    ## define the subfolder that corresponds to the specific parsing parameters of the model
    model_path = mode_path / f"{spacy_model}.{'sentencized' if is_sentencized else ''}/"
    if not model_path.exists():
        model_path.mkdir(exist_ok=True)

    return model_path



def pickle_model(mode_name, spacy_model, is_sentencized, deductive_codes):
    global nlp
    global lemmas_excluded_from_lines
    global excluded_rows

    model_path = get_model_path(mode_name, spacy_model, is_sentencized)

    # pickle the stop words changed by the user
    with open(model_path / "stopwords.pickle", "wb") as f:
        pickle.dump((stopped_lemmas, unstopped_lemmas), f, protocol=pickle.HIGHEST_PROTOCOL)

    # pickle the tokens that are excluded in individual lines by the user
    with open(model_path / "excluded_tokens.pickle", "wb") as f:
        pickle.dump(lemmas_excluded_from_lines, f, protocol=pickle.HIGHEST_PROTOCOL)

    # pickle the rows that are completely excluded by the user
    with open(model_path / "excluded_rows.pickle", "wb") as f:
        pickle.dump(excluded_rows, f, protocol=pickle.HIGHEST_PROTOCOL)

    # pickle the user selected deductive codes
    with open(model_path / "assigned_deductive_codes.pickle", "wb") as f:
        pickle.dump(deductive_codes, f, protocol=pickle.HIGHEST_PROTOCOL)

    # pickle the user actions log
    with open(model_path / "user_actions.pickle", "wb") as f:
        pickle.dump(user_actions, f, protocol=pickle.HIGHEST_PROTOCOL)



def unpickle_defaults_and_model(mode_name, spacy_model, is_sentencized):

    flush_globals()

    global deductive_code_definitions
    global excluded_rows
    global lemmas_excluded_from_lines
    global stopped_lemmas
    global unstopped_lemmas
    global user_actions

    model_path = get_model_path(mode_name, spacy_model, is_sentencized)

    # load deductive code definitions
    with open(CONFIG_FOLDER / "deductive_label_definitions.toml", "rb") as f:
        deductive_code_definitions = tomllib.load(f)

    # load the default stopwords list
    with open(CONFIG_FOLDER / "default_stopwords.pickle", "rb") as f:
        stopped_lemmas = pickle.load(f)

    # load the user-made changes to the stopwords (if they exist)
    stopwords_file = model_path / "stopwords.pickle"
    if stopwords_file.is_file():
        with open(stopwords_file, "rb") as f:
            stopped_lemmas, unstopped_lemmas = pickle.load(f)

    # load the tokens that were excluded on specific lines by the user
    excluded_tokens_file = model_path / "excluded_tokens.pickle"
    if excluded_tokens_file.is_file():
        with open(excluded_tokens_file, "rb") as f:
            lemmas_excluded_from_lines = pickle.load(f)

    # load the lines that were completely excluded by the user
    excluded_rows_file = model_path / "excluded_rows.pickle"
    if excluded_rows_file.is_file():
        with open(excluded_rows_file, "rb") as f:
            excluded_rows = pickle.load(f)

    # load the deductive codes selected by the user
    assigned_codes_file = model_path / "assigned_deductive_codes.pickle"
    if assigned_codes_file.is_file():
        with open(assigned_codes_file, "rb") as f:
            deductive_codes = pickle.load(f)

    # load the user actions log
    user_actions_log_file = model_path / "user_actions.pickle"
    if user_actions_log_file.is_file():
        with open(user_actions_log_file, "rb") as f:
            user_actions = pickle.load(f)

    return deductive_codes


# ---- NLP FUNCTIONS ----

def parse_raw_text(txt: str,
                   timestamp=False,
                   is_interviewer=False,
                   in_sentences=True,
                   use_nlp_tags=False,
                   # resolve_corefs=False
                   ):

    global excluded_rows
    global lemmas_excluded_from_lines
    global nlp
    global graphed_tokens_changed

    first_parse = True if len(lemmas_excluded_from_lines) == 0 else False

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

        # doc = nlp(utterance.strip(), component_cfg={"fastcoref": {'resolve_text': True}}) if resolve_corefs else nlp(utterance.strip())
        doc = nlp(utterance.strip())
        # print("--coref spans: ", doc._.coref_clusters)

        ## TODO -> Move resolved text in a separate table grid column
        ##          Currently, it replaces the existing text (for the sake of quick implementation)

        if in_sentences:
            for s in doc.sents:

                excluded_in_row = lemmas_excluded_from_lines.get(i, [])

                # if the user wants to filter out tokens based on NLP tags
                #    but only if this transcript is being loaded for the first time
                #    otherwise, don't overwrite user-made changes
                if use_nlp_tags and first_parse:
                    excluded_in_row.extend([ t.lemma_ for t in s if nlf.has_excluded_nlp_tag(t) and not t.is_stop ])

                # add the tokens excluded by the algorithm to the rest of exclusions
                if i in lemmas_excluded_from_lines.keys():
                    lemmas_excluded_from_lines[i].extend(excluded_in_row)
                else:
                    lemmas_excluded_from_lines[i] = excluded_in_row

                # remove duplicate elements
                lemmas_excluded_from_lines[i] = list(set(lemmas_excluded_from_lines[i]))

                # create the row data to pass to the ag-grid
                sent_row = row.copy()
                i += 1
                sent_row['line'] = i
                sent_row['in?'] = False if i in excluded_rows else True

                # check if this sentence contains any resolved coreferences
                # and replace the resolved string (for now)

                # print("++sentence spans: ", s.start_char, s.end_char, " >> ", s.text_with_ws)

                utterance = s.text


                ## Umit deactivated coreference resolution on 04/14/2025
                ##      to avoid accidentally leaving it on
                ##      because it slows down the algorithm quite a bit

                # if resolve_corefs:
                #     for c in doc._.coref_clusters:
                #         print("checking cluster: ", c[1])
                #         if s.start_char <= c[1][0] <= s.end_char:
                #             # print("!!! this sentence has a coref !!!")
                #
                #             reference = doc.char_span(c[0][0], c[0][1]).text
                #             pronoun = doc.char_span(c[1][0], c[1][1]).text
                #
                #             # very terrible coding in the line below :)
                #             # TODO -> fix this replace algorithm because it may replace the wrong pronoun
                #             utterance = utterance.replace(pronoun, reference)
                #
                #             # print(" >> new sentence >>", utterance)

                sent_row['utterance'] = utterance
                data.append(sent_row)

        else:
        # here would I go through and make each token bold using markdown?
            i += 1
            row['line'] = i
            row["utterance"] = utterance.strip()
            row['in?'] = False if i in excluded_rows else True
            data.append(row)

    graphed_tokens_changed = True

    return data


def generate_code_checkboxes(line_num, assigned_codes):

    global deductive_code_definitions

    line = str(line_num) ## Umit's note: I noticed that the saved deductive codes loaded with str indexes (04/22/2025)

    ## Create an empty list of values if the user did not select any values for this line
    if line not in assigned_codes.keys():
        assigned_codes[line] = dict(
            (category, "") for category in deductive_code_definitions.keys()
        )

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
                                "index": f"{line}-{category}"
                            },
                            options=[{"label": code, "value": code} for code in deductive_code_definitions[category].keys()],
                            label_checked_class_name="text-success",
                            value = assigned_codes[line][category],
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
                                                        html.Code(deductive_code_definitions[category][code]['keywords'])
                                                    ), className="ms-2",
                                                ),
                                                html.P([
                                                        html.Span("Conceptual Example: ", className="fw-medium"),
                                                        html.Br(),
                                                        html.Em(deductive_code_definitions[category][code]['conceptual_example'])
                                                    ], className="ms-2",
                                                ),

                                                html.P([
                                                        html.Span("Verbatim Excerpt: ", className="fw-medium"),
                                                        html.Br(),
                                                        html.Em(f"\"{deductive_code_definitions[category][code]['verbatim_excerpt']}\"")
                                                    ], className="ms-2",
                                                ),
                                            ],
                                        ) for code in deductive_code_definitions[category].keys()
                                    ],
                                    className="mb-4"
                                )
                            ],
                            target={
                                "type": "code-checklist",
                                "index": f"{line}{category}"
                            },
                            placement="left",
                            trigger="hover",
                            # delay = {"show": 100, "hide": 20}  # leaving here in case we need to activate a delay in the future
                        )
                    ], width=12
                ),
            ], class_name="my-3") for category in deductive_code_definitions.keys()
        ],
        id="code-checkboxes-container",
    )

    return checkboxes_container

# mapping use of certain "tokens" --> words?
def process_utterance(raw_text, row):

    global nlp
    global lemmas_excluded_from_lines

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
                    color="light" if nlp.vocab[token.lemma_].is_stop else "danger" if token.lemma_ in lemmas_excluded_from_lines.get(row, []) else "success",
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
    global lemmas_excluded_from_lines

    row = line["line"] - 1
    doc = nlp(line["utterance"])
    line["highlighted utterance"] = "".join(t.text_with_ws if nlp.vocab[t.lemma].is_stop
                                                              or t.lemma_ in lemmas_excluded_from_lines.get(row, [])
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
    global lemmas_excluded_from_lines

    new_G = nx.Graph()

    # if showing a cumulative graph (start == 0), generate nodes for just until that point
    #    otherwise, generate nodes for the entire transcript

    data_dict_list = data[start:end]     # TODO -> Using the ag_grid's rowData property and filtering start to end instead of refiltering entire dataset over and over again

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
                                                    and not t.lemma_ in lemmas_excluded_from_lines.get(row, [])
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
        tokens_in_vocab = [t for t in new_G.nodes if not nlp.vocab[t].is_oov]

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
    global graphed_tokens_changed

    # UA > if any edits were made in the utterance table or line number, regenerate the graph (nodes and the edge matrix)
    #       otherwise use the same graph for visualization changes
    if graphed_tokens_changed:
        # now let's generate the knowledge graph
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
        graphed_tokens_changed = False

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
    edge_x = []
    edge_y = []

    light_edge_x = []
    light_edge_y = []

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
    
            graph_metrics = dcc.Graph(figure=fig_metrics, config=graph_config_options)
    
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
        fig_metrics.update_layout(
            showlegend=False,
            title=dict(
                text=mode_name,
                font=dict(size=18, weight="bold"),
                subtitle=dict(
                    text=f"density = {nx.density(G):.3f} | {subtitle_user_choices}",
                    font=dict(size=10, color="gray")
                ),
                x=0.5,
                y=0.99,
                xanchor="center",
                yanchor="top"
            ),
            margin = dict(l=0, r=0, t=80, b=40),
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
    file_list, id="input-file-dropdown", value="_demo_cory1_abc.txt", persistence=True,
)

model_name_input = dbc.Input(id="mode-name", value="", placeholder="Enter model name ...")

raw_text_input = dbc.Textarea(
    placeholder="Copy and paste some text here.", value="", rows=10, id="raw-text"
)

parse_button = dbc.Button("Parse", id="parse-button", size="lg", n_clicks=0)

sentencize_checkbox = dbc.Checkbox(label="Split into sentences?", id="by-sent", value=True, persistence=True)
apply_tags_checkbox = dbc.Checkbox(label="Use NLP tags to infer irrelevant tokens", id="use-nlp-tags", value=True, persistence=True)
#corefs_checkbox = dbc.Checkbox(label="Resolve coreferences", id="resolve-corefs", value=False, disabled=False if heroku_access_pwd is None else True)

model_selection_dropdown = dbc.Select(
    id="model-selection-dropdown",
    options=[
        {"label": "Small", "value": "en_core_web_sm"},
        {"label": "Medium", "value": "en_core_web_md"},
        {"label": "Large", "value": "en_core_web_lg"},
    ],
    persistence=True,
    value="en_core_web_lg"
)

reset_button = dbc.Button(
    "Reset Model",
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
        {"label": "Highlight Included Tokens", "value": 3},
    ],
    value=[0, 1, 2],
    inline=True,
    class_name="mb-4",
    id="inclusion-options",
    persistence=True,
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
                            model_name_input
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
                    ),
                    class_name="mb-4",
                ),
                dbc.Row(

                ),
                dbc.Row([
                    dbc.Col(sentencize_checkbox, xl=2),
                    dbc.Col(apply_tags_checkbox, xl=3),
                    # dbc.Col(corefs_checkbox, xl=2),
                    dbc.Col(width=2),
                    dbc.Col(
                        dbc.InputGroup([
                            dbc.InputGroupText("Model"),
                            model_selection_dropdown
                        ]),
                        xl=3
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

empty_utterances_table_data = [{
    'line': '0',
    'time': '00:00:00',
    'speaker': 'N/A',
    'utterance': 'Processed text will be displayed in this table.',
    'in?': False
}]

utterances_accordion = dbc.Accordion(
    dbc.AccordionItem(
        [inclusion_options,
            html.Div(
                [
                    generate_utterance_table(empty_utterances_table_data, (0, 1, 2), False)
                ],
                id="utterances-div",
            )
        ],
        id="revise",
        title="Revise and Code",
    ),
    # active_item="1",  # collapsed by default
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
                    dbc.Input(id="min-similarity", type="number", min=0, max=1, step=0.1, value=0.8, disabled=False, persistence=True),
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
        html.H4("Graph Construction", className="my-4"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Weak min co-occurrence"),
                        dbc.Input(
                            id="min-co",
                            type="number",
                            min=1,
                            max=10,
                            step=1,
                            value=1,
                            persistence=True
                        ),
                    ]),
                    class_name="mt-2",
                    md=6, lg=4, xl=3,
                ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Strong min co-occurrence"),
                        dbc.Input(
                            id="min-strong-co",
                            type="number",
                            min=1,
                            max=10,
                            step=1,
                            value=2,
                            persistence=True
                        )]
                    ),
                    class_name="mt-2",
                    md=6, lg=4, xl=3,
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
                            persistence=True
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
                        dbc.Input(id="layout-iterations", type="number", min=0, max=500, step=1, value=10, persistence=True),
                    ]),
                    lg=4,
                    xl=3,
                ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Spring k"),
                        dbc.Input(id="layout-k", type="number", min=0, max=100, step=0.05, value=0.5, persistence=True),
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
    className="border rounded p-4 my-4",
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
                        ## Umit commented out the following lines on 02/24/2025 to deactivate
                        ##      the treemap visualization of token counts

                        # dbc.Row(
                        #     dbc.Col(
                        #         [
                        #             html.H4("Frequency map"),
                        #             html.Div(
                        #                 "Something must have gone wrong!",
                        #                 id="utterance-stats",
                        #             ),
                        #         ]
                        #     ),
                        # ),

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


# assigned_deductive_codes = dict()  ## keeps the labels selected by the user for each line
# deductive_code_definitions = dict()  ## Keeps the info about the labels, not user selections
# excluded_rows = set()
# graph_button_clicked = False
# graphed_tokens_changed = False  ## TODO: problematic global because once it's set to True, it remains True.
# lemmas_excluded_from_lines = dict()
# stopped_lemmas = set()
# unstopped_lemmas = set()
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
        dcc.Store(id="modal-row-id"), # to keep track of the id of the row that is being revised in the modal view
        dcc.Store(id="parsed-data"),
        dcc.Store(id="assigned-deductive-codes", data=dict()),
        dbc.Row(dbc.Col(utterances_accordion)),
        dbc.Row(dbc.Col(generate_div)),
        dbc.Row(dbc.Col(graph_view_options_div)),
        dbc.Row(dbc.Col(metrics_viewer_wrapper_div)),
        dbc.Row(dbc.Col(user_log_accordion)),
        coding_modal
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
def load_input_file_callback(file_name: str):
    if file_name == "__manual entry__":
        return "", ""

    # gets path to file and removed .txt from the file's name
    file_path = Path(INPUT_FOLDER) / file_name
    model_name = file_name.removesuffix(".txt")

    # checks file existence
    if not file_path.is_file():
        return "It doesn't seem like that file exists anymore.", model_name

    # opens file and reads the file 
    # puts the text in one string instead of a list of lines
    with open(file_path, "r") as f:
        file_text = "".join([f"{line.strip()}\n" for line in f.readlines()])

    # checks if there is actually text (rather than empty file/string)
    if len(file_text) > 0:
        return file_text, model_name

    return "File was there, but it had no text.", model_name


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
    State("by-sent", "value"),
    State("model-selection-dropdown", "value"),
    prevent_initial_call=True,
)
def reset_model_button_callback(n_reset_clicks, name, sentencized, model):

    if ctx.triggered_id == "reset-button":

        # first, let's get rid of the existing user generated model files

        ## path of current model folder
        model_path = Path(f"./models/{str(name).strip()}-{model}-sent_{sentencized}/")

        ## checks whether a model folder exists
        if model_path.is_dir():

            # first, delete all the files in the folder (unlink == delete)
            for pickle_file in model_path.iterdir():
                pickle_file.unlink()

            ## then, remove the folder itself
            model_path.rmdir()

        flush_globals()

        # now let's trigger a page refresh

        return dbc.Alert(
            [
                html.I(className="bi bi-info-circle-fill me-2"),
                dbc.Badge(
                    f"<{str(name).strip()}-{model}-sent_{sentencized}>",
                    color="light",
                    text_color="danger",
                    class_name="p-2"
                ),
                html.Span(" was reset successfully; A page refresh is highly recommended."),
            ],
            color="danger"
        )

    else:
        return ""



@app.callback(
    Output("data-table", "rowData"),
    Output("input-accordion", "active_item"),
    Output("graph-button", "disabled"),
    Output("parsed-data", "data"),
    Output("assigned-deductive-codes", "data", allow_duplicate=True),

    Input("parse-button", "n_clicks"),

    State("mode-name", "value"),
    State("raw-text", "value"),
    State("by-sent", "value"),
    State("model-selection-dropdown", "value"),
    State("use-nlp-tags", "value"),
    State("modal-row-id", "data"),
    State("data-table", "rowData"),
    # State("resolve-corefs", "value"),

    prevent_initial_call=True,
)
def parse_button_callback(
        n_parse_button_clicks,
        name,
        txt,
        sentencized,
        spacy_model,
        use_nlp_tags,
        revised_row_id,
        existing_row_data,
        # resolve_corefs,
):
    global user_actions
    global deductive_code_definitions
    global excluded_rows
    global lemmas_excluded_from_lines
    global nlp
    global stopped_lemmas
    global graphed_tokens_changed
    global unstopped_lemmas


    # first, reset all the globals
    #   to make sure that switching between transcripts doesn't mess things up
    flush_globals()
    excluded_rows = []

    deductive_codes = unpickle_defaults_and_model(name, spacy_model, sentencized)

    # reload the model because it only pulls default stopwords if loaded from the beginning
    nlp = spacy.load(spacy_model, exclude=["ner"])

    # if resolve_corefs:
    #     nlp.add_pipe("fastcoref", config={  'device': 'cpu',
    #                                                 # 'model_architecture': 'LingMessCoref', # this model runs slower
    #                                                 # 'model_path': 'biu-nlp/lingmess-coref' # comment these two lines if you want the default faster model
    #                                              })

    # update stop_words of the small model
    #   I have to do it this y because spacy's to_disk method doesn't save stopwords
    for word in stopped_lemmas:
        nlp.vocab[word].is_stop = True

    for word in unstopped_lemmas:
        nlp.vocab[word].is_stop = False


    # tokens that are excluded from a specific line, but not the entire analysis
    time = True
    interviewer = True

    # here in possible changes
    parsed_data = parse_raw_text(
        txt, timestamp=time,
        is_interviewer=interviewer,
        in_sentences = sentencized,
        use_nlp_tags = use_nlp_tags,
    )

    return generate_highlighted_utterances(parsed_data), "1", False, parsed_data, deductive_codes


# needs to filter out interviewers as third option
@app.callback(
    Output('data-table', 'columnState'),
    Output('data-table', 'dashGridOptions'),
    Input("inclusion-options", "value"),
    Input("coding-modal", "is_open"),   # TODO -> separate this input into a new callback and have it update the parsed data
    State("parsed-data", "data")
)
def apply_table_layout_filters_callback(table_display_options, coding_modal_was_open, parsed_data):

    new_state = [
        {'colId': 'line'},
        {'colId': 'time', 'hide': 0 not in table_display_options},
        {'colId': 'speaker', 'hide': 1 not in table_display_options},
        {'colId': 'utterance', 'hide': 3 in table_display_options, 'flex':1},
        {'colId': 'highlighted utterance', 'hide': 3 not in table_display_options, 'flex':1},
        {'colId': 'in?'},
    ]

    new_filter = {
        'isExternalFilterPresent': {'function': 'false'}
    }
    if 2 in table_display_options:
        new_filter = {
            'isExternalFilterPresent': {'function': 'true'},
            'doesExternalFilterPass': 
                {'function': "params.data.speaker != 'Interviewer'"}
        }

    return new_state, new_filter


@app.callback(
    Output("token-buttons", "children"),
    # Output("utterance-stats", "children"), # umit temporarily commented out this line on 02/24/2025 to deactivate the treemap visualization
    Output("code-checkboxes-container", "children"),
    Output("coding-modal", "is_open"),
    Output("modal-row-id", "data"),
    Input("data-table", "cellClicked"),
    Input({"type": "toggle-token", "index": ALL, "stop": ALL}, "n_clicks"),
    State("data-table", "rowData"),
    State("assigned-deductive-codes", "data"),
    prevent_initial_call=True,
)
def revise_tokens_view_callback(cell, toggle_clicks, row_data, assigned_codes):
    global graphed_tokens_changed
    global lemmas_excluded_from_lines
    global user_actions
    # create global table 
    if cell is not None:

        row = int(cell["rowId"])
        graphed_tokens_changed = False

        if len(toggle_clicks) > 0:

            if 1 in toggle_clicks:

                toggled_token = ctx.triggered_id["index"]
                was_stop = ctx.triggered_id["stop"]

                # to log the time this token was toggled
                curr_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                if was_stop:

                    nlp.vocab[toggled_token].is_stop = False
                    stopped_lemmas.discard(toggled_token)
                    unstopped_lemmas.add(toggled_token)
                    user_actions.append({'time': curr_time, 'line': row, 'change': f'\"{toggled_token}\" was toggled ON.\n'})
                    # change_log.append(html.P(f'At time {curr_time}: \"{toggled_token}\" was toggled ON.\n'))

                else:
                    # if a token was not a stop word, first check if it is in the excluded tokens list

                    if toggled_token in lemmas_excluded_from_lines.get(row, []):
                        # if it was an excluded token, turn it into a stop word
                        # and remove it from the exluded tokens list

                        nlp.vocab[toggled_token].is_stop = True
                        stopped_lemmas.add(toggled_token)
                        unstopped_lemmas.discard(toggled_token)
                        lemmas_excluded_from_lines[row].remove(toggled_token)
                        user_actions.append({'time': curr_time, 'line': row, 'change': f'\"{toggled_token}\" was toggled OFF.\n'})
                        # change_log.append(html.P(f'At time {curr_time}: \"{toggled_token}\" was toggled OFF.'))

                    else:

                        # if it was not in excluded tokens list, turn it into an excluded token

                        if len(lemmas_excluded_from_lines.get(row, [])) == 0:
                            lemmas_excluded_from_lines[row] = [toggled_token]
                        else:
                            lemmas_excluded_from_lines[row].append(toggled_token)

                        user_actions.append({'time': curr_time, 'line': row, 'change': f'\"{toggled_token}\" was excluded from the line.'})
                        # change_log.append(html.P(f'At time {curr_time}: \"{toggled_token}\" was excluded from line {row + 1}.'))

                graphed_tokens_changed = True

        token_buttons = process_utterance(row_data[row]["utterance"], row=row)

        codes = generate_code_checkboxes(row, assigned_codes)

        return token_buttons, codes, True, row
    else:
        return "Something", "went wrong", False, -1


@app.callback(
    Output("graph-div", "children"),
    Output("graph-slider", "marks"),
    Output("graph-slider", "value"),
    Output("metrics-div", "children"),
    Output("min-strong-co", "value"),
    Output("log-data-table", "rowData"),

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

    State("parsed-data", "data"),
    State("inclusion-options", "value"),
    State("use-deductive-codes", "value"),
    State("assigned-deductive-codes", "data"),
    State("graph-button", "disabled"),
    State("mode-name", "value"),
    State("combine-by-similarity", "value"),
    State("min-similarity", "value"),
    State('data-table', 'virtualRowData'),
    State("by-sent", "value"),
    State("model-selection-dropdown", "value"),

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
        # toggled_token,
        parsed_data,
        selected_inclusion_options,
        use_deductive_codes,
        assigned_codes,
        graph_button_disabled,
        mode_name,
        combine_by_similarity,
        min_similarity,
        active_row_data,
        is_sentencized,
        spacy_model
):

    global graphed_tokens_changed
    global graph_button_clicked
    global user_actions

    empty_return = ["You need to process some data.", {0: 'N/A'}, [0, 0], "You need to process some data.", min_co_occurrence, user_actions]

    if graph_button_disabled:
        return empty_return

    if ctx.triggered_id == "graph-button":
        graph_button_clicked = True
        graphed_tokens_changed = True

        # also save (pickle) the user's work if the user clicks the "Generate Knowledge Graph" button
        pickle_model(mode_name, spacy_model, is_sentencized, assigned_codes)

    if ctx.triggered_id == "graph-slider":
        graphed_tokens_changed = True

    if ctx.triggered_id == "min-co":
        graphed_tokens_changed = True

    if ctx.triggered_id == "min-strong-co":
        graphed_tokens_changed = True

    if ctx.triggered_id == "inclusion-options":
        graphed_tokens_changed = True
    
    if not graph_button_clicked:
        return empty_return

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
    list_of_marks = sorted([l['line'] for l in active_row_data if l['in?']])
    slider_marks = {r: '' for r in list_of_marks}

    # determine the start and end of the range that the user picked
    start = selected_range[0]
    end = selected_range[1]
    if ctx.triggered_id == "graph-button":
        last_line = list(slider_marks.keys())[-1]
        end = selected_range[1] if selected_range[1] != 0 and selected_range[1] <= last_line else last_line

    end = end if end < len(parsed_data) else len(parsed_data)

    selected_range = list([start, end])

    graph, stats = draw_token_graph_plotly_object(
        data=parsed_data,
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
        show_weak_links = display_weak_links,
        combine_by_similarity=combine_by_similarity,
        min_similarity=min_similarity,
    )
    # change_log = [{'dict a': 'test a'}, {'dict b': 'test b'}, {'dict c': 'test c'}]
    return graph, slider_marks, selected_range, stats, min_strong_co_occurrence, user_actions
    # need to update change_log


@app.callback(
    Output("assigned-deductive-codes", "data", allow_duplicate=True),
    Input({"type": "code-checklist", "index": ALL}, "value"),
    State("assigned-deductive-codes", "data"),
    prevent_initial_call=True
)
def update_user_assigned_deductive_codes_callback(clicked_checkboxes, assigned_codes):

    line_num, category = ctx.triggered_id["index"].split("-")
    line_num = line_num.strip()

    if line_num not in assigned_codes: assigned_codes[line_num] = dict()

    assigned_codes[line_num][category] = ctx.triggered[0]["value"]

    return assigned_codes


@app.callback(
    Output("log-data-table", "rowData", allow_duplicate=True),
    Input("data-table", "cellValueChanged"),
    State("parsed_data", "data"),
    prevent_initial_call=True,
)
def update_included_lines_callback(changed, parsed_data):
    global graphed_tokens_changed
    global user_actions

    if changed:
        i = int(changed[0]["rowId"])
        cell_incl = changed[0]['data']['in?']
        parsed_data[i]['in?'] = cell_incl
        graphed_tokens_changed = True
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