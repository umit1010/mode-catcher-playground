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
from plotly.subplots import make_subplots

# ---- PLATFORM ----

nlp = None

stopped_words = set()
unstopped_words = set()

assigned_codes = dict()

stored_data = None

theoretical_code_list = [
    'emergent',
    'centralized',
    'probabilistic',
    'deterministic',
    'feedback',
    'fitting',
    'levels',
    'slippage',
]

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)


# ---- NLP ----


def parse_raw_text(txt: str,
                   timestamp=False,
                   speaker=False,
                   interviewer=False):
    input_lines = [line.strip().replace('\n', '')
                   for line in txt.splitlines()
                   if len(line.strip()) > 0 and line.count(':') > 2]

    # parse the text line by line
    re_time_splitter = re.compile(r'(\[[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\])')

    data = []

    if not interviewer:
        input_lines = [line for line in input_lines if line.lower().count('interviewer') == 0]

    for i, line in enumerate(input_lines):

        _, time, speaker_speech = re_time_splitter.split(line)
        speaker, utterance = speaker_speech.strip().split(':')
        speaker = str(speaker)

        row = {'line': i + 1}

        if timestamp:
            row['time'] = time

        if speaker:
            row['speaker'] = speaker

        row['utterance'] = utterance.strip()

        data.append(row)

        if i not in assigned_codes.keys():
            assigned_codes[i] = [False] * len(theoretical_code_list)

    return data


def generate_code_checkboxes(line_num, values=None):
    if values is not None:
        assigned_codes[line_num] = values

    container = html.Div(
        [
            html.Div([
                dbc.Checkbox(label=code[1],
                             value=assigned_codes[line_num][code[0]],
                             id={'type': 'code-checkbox', 'index': code[1]})
            ], className='w-50') for code in enumerate(theoretical_codes_list)
        ],
        className='d-flex align-content-start flex-wrap',
        id='code-checkboxes-container'
    )
    return container


def process_utterance(raw_text):

    global nlp

    doc = nlp(raw_text.strip().lower())

    all_tokens = [token.lemma_
                  for token in doc
                  if not nlp.vocab[token.lemma].is_stop and not nlp.vocab[token.lemma].is_punct
                  ]

    token_counts = Counter(all_tokens)

    data_dict = {'token': list(token_counts.keys()), 'count': list(token_counts.values())}

    df = pd.DataFrame.from_dict(data_dict)

    buttons_for_text = html.Div([
        html.Span(dbc.Button(
            token.text,
            id={'type': 'toggle-token', 'index': token.lemma_, 'stop': True},
            n_clicks=0,
            color='light',
            class_name='m-1',
            size='sm')) if nlp.vocab[token.lemma].is_stop
        else html.Span(token.text, className='mx-1') if nlp.vocab[token.lemma].is_punct
        else html.Span(dbc.Button(token.text,
                                  id={'type': 'toggle-token', 'index': token.lemma_, 'stop': False},
                                  n_clicks=0,
                                  color='warning',
                                  class_name='m-1',
                                  size='sm'))
        for token in doc
    ])

    fig = px.treemap(
        df,
        path=[px.Constant('tokens'), 'token'],
        values='count',
        color='count',
        hover_data='token',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=df['count'].mean()
    )

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    token_treemap = dcc.Graph(figure=fig,
                              responsive=True,
                              style={'height': '200px'})

    return buttons_for_text, token_treemap


def pickle_model(mode_name):

    global nlp

    models_folder = Path('./models/')
    models_folder.mkdir(exist_ok=True)

    mode_folder = models_folder / mode_name
    mode_folder.mkdir(exist_ok=True)

    stopwords_file = mode_folder / 'stopwords.pickle'
    theoretical_codes_file = mode_folder / 'theoretical_codes.pickle'

    with open(stopwords_file, 'wb') as swf:
        pickle.dump((stopped_words, unstopped_words),
                    swf,
                    protocol=pickle.HIGHEST_PROTOCOL)

    with open(theoretical_codes_file, 'wb') as tcf:
        pickle.dump(assigned_codes,
                    tcf,
                    protocol=pickle.HIGHEST_PROTOCOL)


def process_tokens():

    return "I did not implement this yet"


def generate_graph(start_line=0,  # if > 0, dmc mode is activated
                   end_line=1,
                   case_name="",
                   with_codes=False,
                   layout_iterations=3,
                   min_co=1,
                   min_dmc_co=2,
                   node_size_multiplier=2):

    global nlp

    # if showing a cumulative graph, generate nodes for just until that point
    #    otherwise, generate nodes for the entire transcript
    data_dict_list = stored_data[0:end_line] if start_line == 0 else stored_data

    # if showing a cumulative graph, just show the current line number
    #    otherwise, show the range of the line numbers
    current_lines = f'{end_line}' if start_line == 0 else f'[{start_line},{end_line}]'

    # create an empty dataframe that will be filled in the loop
    df = pd.DataFrame()

    # next, iterate over each line's unique tokens and add them to the matrix
    for line in data_dict_list:

        processed_line = nlp(line['utterance'].strip().lower())

        line_tokens = list(set([token.lemma for token in processed_line
                                if not nlp.vocab[token.lemma].is_punct
                                and not nlp.vocab[token.lemma].is_stop
                                and not token.is_punct
                                and not token.is_stop]))

        # # append the theoretical codes to the list of tokens in the line
        # if with_codes:
        #     selections = assigned_codes[line['line'] - 1]
        #     codes_to_include = [theoretical_code_list[i] for i in range(len(theoretical_code_list)) if selections[i]]
        #     code_tokens = [nlp.vocab.strings[c] for c in codes_to_include]
        #     line_tokens.extend(code_tokens)

        # first loop is iterating over each token
        # second loop iterates over the tokens after the current token
        for i in range(len(line_tokens)):

            row = line_tokens[i]

            # if the token was not added to the dataframe yet, add it as 1 count
            #    if it already exists, increment it by 1 (count)
            if row not in df.index or row not in df.columns:
                df.loc[row, row] = 0
            else:
                df.loc[row, row] += 1

            for j in range(i + 1, len(line_tokens)):

                col = line_tokens[j]

                if col == row: continue

                # if the token was not added to the dataframe yet, add it as 1 count
                #    if it already exists, increment it by 1 (count)
                if col not in df.index or col not in df.columns:
                    df.loc[col, col] = 0
                # else:
                #     df.loc[col, col] += 1

                # Gotta check for NaN because otherwise NaN + 1 = NaN.
                if np.isnan(df.loc[col, row]):
                    df.loc[col, row] = 0
                    df.loc[row, col] = 0

                # if no window is provided,
                #    just count as usual
                # if we have a window, then we will only calculate counts and co-occurrences
                #    between nodes within that window
                if start_line == 0:

                    df.loc[row, col] += 1
                    df.loc[col, row] += 1

                elif start_line <= line['line'] <= end_line:

                    df.loc[row, col] += 1
                    df.loc[col, row] += 1

    # ---
    # CREATE THE GRAPH
    # ---

    node_frequencies = [df.loc[token, token] for token in df.index]

    nodes = [(token, {'weight': df.loc[token, token], 'label':nlp.vocab.strings[token]}) for token in df.index]

    # create the network
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for i in range(len(df.index)):
        row = df.index[i]
        for j in range(i + 1, len(df.index)):
            col = df.index[j]
            connections = df.loc[row, col]
            if connections >= min_co:  # I add connections even if two tokens co-occur once, but don't show them
                G.add_edge(row, col, weight=connections)

    # calculate the metrics
    node_degrees = [G.degree[token] for token in df.index]
    node_clustering = nx.clustering(G)
    ave_clustering = nx.average_clustering(G) if len(node_clustering) > 0 else 0

    # create representation attributes (size, color, text, etc.)
    #   I add 1 to node size because the ones with degree 0 disappear
    node_sizes = [(n * node_size_multiplier) for n in node_frequencies]
    node_colors = [(G.degree[token] + 1) for token in df.index]

    node_labels = list(nx.get_node_attributes(G, 'label').values())

    hover_texts = [f'<b>{node_labels[idx]}</b> <br> '
                   f'𝑓: {node_frequencies[idx]} <br> '
                   f'deg: {node_degrees[idx]} <br>'
                   f'clustering: {node_clustering[token]:.3f} <br>'
                   for (idx, token) in enumerate(df.index)]

    # generate a spring layout for node locations
    layout_seed = np.random.RandomState(42)
    pos = nx.spring_layout(G, iterations=layout_iterations, seed=layout_seed, scale=4)
    # pos = nx.random_layout(G,seed=layout_seed)    # comment out to test random
    # pos = nx.circular_layout(G)                   # comment out to test random

    # create the plotly graph for the network
    edge_x = []
    edge_y = []

    for edge in G.edges():
        edge_attr = G.get_edge_data(edge[0], edge[1], default={'weight': 1})
        if edge_attr['weight'] >= min_dmc_co:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # create a representation of the connections among tokens
    # that co occur more than min_co but less than min_dmc_co
    if min_co < min_dmc_co:

        light_edge_x = []
        light_edge_y = []

        for edge in G.edges():
            edge_attr = G.get_edge_data(edge[0], edge[1], default={'weight': 1})
            if edge_attr['weight'] <= min_co:  # only show link if co-occur > cutoff
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                light_edge_x.append(x0)
                light_edge_x.append(x1)
                light_edge_x.append(None)
                light_edge_y.append(y0)
                light_edge_y.append(y1)
                light_edge_y.append(None)

        light_edge_trace = go.Scatter(
            x=light_edge_x, y=light_edge_y,
            line=dict(width=0.1, color='#AAA'),
            hoverinfo='none',
            mode='lines'
        )
    else:
        light_edge_trace = go.Scatter()

    node_x = [pos[i].tolist()[0] for i in pos]
    node_y = [pos[i].tolist()[1] for i in pos]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hovertext=hover_texts,
        hoverinfo='text',
        text=node_labels,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='Portland',
            reversescale=False,
            color=node_colors,
            size=node_sizes,
            line_width=1,
            colorbar=dict(title=dict(text='degree'))
        )
    )

    type_label = "Cumulative" if start_line == 0 else "DMC"

    fig_graph = go.Figure(
        data=[light_edge_trace, edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"{case_name} | {type_label} View @ at {current_lines}",
                x=0.5,
                xanchor='center'
            ),
            font=dict(
                size=16
            ),
            hovermode='closest',
            height=600,
            margin=dict(l=0, r=0, t=40, b=40),
            showlegend=False
        )
    )

    fig_graph.update_xaxes(showticklabels=False)
    fig_graph.update_yaxes(showticklabels=False)

    graph_network = dcc.Graph(figure=fig_graph, config={'displayModeBar': True})

    # metrics plots

    graph_metrics = "Nothing to display yet!"

    # first create a sorted list of degrees for plotting as a scatter plot and histogram
    connected_nodes = dict(
        [(node_labels[idx], degree)
         for (idx, degree) in enumerate(node_degrees)
         if degree > 0]
    )

    # only attempt to plot if there are any tokens with degree higher than 0
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
            [(node_labels[idx], node_clustering[key])
             for (idx, key) in enumerate(node_clustering)
             if node_clustering[key] > 0]
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

        fig_metrics.update_yaxes(range=[0, 60], row=1, col=1)
        fig_metrics.update_yaxes(range=[0, 1.1], row=1, col=2)
        fig_metrics.update_layout(showlegend=False,
                                  title=dict(
                                      text=f"density = {nx.density(G):.3f}",
                                      x=0.5, xanchor='center'),
                                  margin=dict(l=0, r=0, t=40, b=40)
                                  )

    return graph_network, graph_metrics


# ---- INTERFACE ----

# -- input section --

INPUT_FOLDER = 'samples'

input_folder_path = Path(INPUT_FOLDER)

file_list = ['__manual entry__']

if input_folder_path.is_dir():
    text_files = [f.name for f in sorted(input_folder_path.glob('*.txt'))]
    if len(text_files) > 0:
        file_list.extend(text_files)

input_file_dropdown = dbc.Select(
    file_list,
    id='input-file-dropdown',
    value='01_cj2.txt'
)

mode_name_input = dbc.Input(id='mode-name',
                            value="",
                            placeholder="Enter mode name ...")

raw_text_input = dbc.Textarea(
    placeholder="Copy and paste some text here.",
    value="",
    rows=10,
    id='raw-text'
)

parse_button = dbc.Button('Parse',
                          id='parse-button',
                          size='lg',
                          n_clicks=0)

reset_button = dbc.Button('Reset Mode',
                          id='reset-button',
                          color='danger',
                          outline=True,
                          class_name='ms-auto',
                          n_clicks=0)

inclusion_options = dbc.Checklist(
    options=[
        {'label': 'Display Timestamp', 'value': 0},
        {'label': 'Display Speaker', 'value': 1},
        {'label': 'Ignore Interviewer Utterances', 'value': 2}
    ],
    value=[2],
    inline=True,
    class_name='mb-4',
    id='inclusion-options'
)

input_accordion = dbc.Accordion(
    [
        dbc.AccordionItem(
            [
                dbc.Row(
                    dbc.Col([
                        dbc.Label('Input File:'),
                        input_file_dropdown,
                        html.P('')
                    ], width=10, lg=6)
                ),
                dbc.Row(
                    dbc.Col([
                        dbc.Label('Mode name:'),
                        mode_name_input,
                        html.P('')
                    ], width=10, lg=6)
                ),
                dbc.Row(
                    dbc.Col([
                        dbc.Label('Transcript:'),
                        raw_text_input,
                    ])
                ),
                dbc.Row(
                    dbc.Col([
                        inclusion_options,
                        parse_button,
                    ], class_name='mt-4'),
                ),
                dbc.Row([
                    dbc.Col([
                        reset_button,
                    ], class_name='d-flex align-items-end')
                ]),
                html.P(''),
                dbc.Row([
                    dbc.Col([
                        html.Div('', id='reset-message-div', className='ms-auto'),
                    ], class_name='d-flex align-items-end')
                ]),

            ],
            title="Input",
            item_id='0')
    ],
    active_item='0',
    id="input-accordion",
    className='my-4'
)

# -- utterances section --

utterances_wrapper_div = html.Div(
    [
        html.H3('Revise and Code', className='mb-4'),
        html.Div(
            [
                html.P('Processed text will be displayed here as a datatable.', className='lead')
            ], id='utterances-div'
        )
    ], className='border rounded p-4 my-4'
)

graph_button = dbc.Button('Generate Graph',
                          id='graph-button',
                          class_name='mt-4',
                          size='lg',
                          n_clicks=0,
                          disabled=True)

generate_div = html.Div(
    [
        graph_button
    ],
    className='border rounded p-4 my-4'
)

theoretical_codes_list = [
    'emergent',
    'centralized',
    'probabilistic',
    'deterministic',
    'feedback',
    'fitting',
    'levels',
    'slippage',
]

code_checkboxes_container = dbc.Container(
    "",
    fluid=True,
    class_name='d-flex align-content-start flex-wrap',
    id='code-checkboxes-container'
)

# -- graph view --

graph_type_row = dbc.Row([

    dbc.Col(
        dbc.Checkbox(label="Theoretical Codes", id='include-codes', value=False),
        xs=12, md=6, xl=2,
        class_name='mt-3'),

    dbc.Col(
        dbc.Checkbox(label="DMC Mode", id='dmc-mode', value=False),
        xs=12, md=6, xl=2,
        class_name='mt-3')

])

grap_options_row = html.Div(
    [
        html.H4("Structure", className='my-4'),
        dbc.Row([

            dbc.Col([
                html.Span('Min Tokens per Line: ', className='me-4'),
                dcc.Input(id='min-tokens',
                          type="number",
                          min=1, max=20, step=1,
                          value=3,
                          style={'margin-top': '-6px'}
                          ),
            ], md=12, xl=3, class_name='d-flex mt-3'),

            dbc.Col([
                html.Span('Min Co-occurrence: ', className='me-4'),
                dcc.Input(id='min-co',
                          type="number",
                          min=1, max=10, step=1,
                          value=2,
                          style={'margin-top': '-6px'}
                          ),
            ], md=12, xl=3, class_name='d-flex mt-3'),

            dbc.Col([
                html.Span('Min DMC Co-occurrence: ', className='me-4'),
                dcc.Input(id='min-dmc-co',
                          type="number",
                          min=1, max=10, step=1,
                          value=2,
                          style={'margin-top': '-6px'}
                          ),
            ], md=12, xl=3, class_name='d-flex mt-3'),

            dbc.Col([
                html.Span('DMC Window: ', className='me-4'),
                dcc.Input(id='dmc-window',
                          type="number",
                          min=1, max=11, step=2,
                          value=3,
                          style={'margin-top': '-6px'}
                          )
            ], md=12, xl=3, class_name='d-flex mt-3'),

        ], class_name='my-4', justify='center'),

        html.H4("Visualization", className='my-4'),

        dbc.Row([

            dbc.Col([
                html.Span('Spring: ', className='me-4'),
                dcc.Input(id='layout-iterations',
                          type="number",
                          min=0, max=50, step=2,
                          value=3,
                          style={'margin-top': '-6px'}
                          )
            ], md=12, xl=3, class_name='d-flex mt-3'),

            dbc.Col([
                html.Span('Node Size: ', className='me-4'),
                dcc.Input(id='node-size',
                          type="number",
                          min=2, max=18, step=1,
                          value=2,
                          style={'margin-top': '-6px'},
                          className='ms-4'
                          ),
            ], md=12, xl=3, class_name='d-flex mt-3'),

        ], class_name='my-4', justify='center')
    ], className='my-4'
)

graph_view_wrapper_div = html.Div(
    [
        html.H3('Knowledge Graph', className='mb-4'),
        html.P(' '),
        graph_type_row,
        html.P(' '),
        html.Div(
            [
                html.P(' '),
                html.P(
                    'Knowledge graph will be displayed once you generate it.',
                    className='lead text-center m-4 p-4'),
                html.P(' ')
            ], id='graph-div'
        ),
        dcc.Slider(id='graph-slider',
                   min=1, max=2, step=1,
                   value=1,
                   marks=None,
                   tooltip={"placement": "bottom", "always_visible": True},
                   className='my-4'),
        grap_options_row
    ], className='border rounded p-4 my-4'
)

metrics_viewer_wrapper_div = html.Div(
    [
        html.H3('Metrics', className='mb-4'),
        html.P(' '),
        html.Div(
            'Will be updated once the graph is generated.',
            id='metrics-div'
        )
    ], className='border rounded p-4 my-4'
)

# -- coding modal view --

coding_modal = dbc.Modal(
    [
        dbc.ModalHeader(
            dbc.ModalTitle('Modify'),
            close_button=True
        ),
        dbc.ModalBody(
            dbc.Row([
                dbc.Col('', id='token-buttons'),
                dbc.Col([
                    dbc.Row(
                        dbc.Col([
                            html.H4('Frequency map'),
                            html.Div(
                                'Something must have gone wrong!',
                                id='utterance-stats'
                            )
                        ]),
                    ),
                    dbc.Row(
                        dbc.Col(
                            [
                                html.H4('Theoretical Codes'),
                                html.P('not yet functional', className='small text-italic'),
                                code_checkboxes_container
                            ]
                        ), class_name='mt-4'
                    )
                ])
            ])
        )], id='coding-modal',
    # fullscreen=True,
    scrollable=True,
    size='xl',
    is_open=False,
    centered=True
)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col([
                html.H1([
                    'mode-catcher ', html.Em('playground')
                ], className='text-center m-4'),
                input_accordion,
            ])
        ),
        dbc.Row(
            dbc.Col(
                utterances_wrapper_div
            )
        ),
        dbc.Row(
            dbc.Col(
                generate_div
            )
        ),
        dbc.Row(
            dbc.Col(
                graph_view_wrapper_div
            )
        ),
        dbc.Row(
            dbc.Col(
                metrics_viewer_wrapper_div
            )
        ),
        coding_modal
    ],
    fluid=True,
    class_name='p-4'
)


# ---- CALLBACKS ----
@app.callback(
    Output('raw-text', 'value'),
    Output('mode-name', 'value'),
    Input('input-file-dropdown', 'value')
)
def load_input_file(file_name: str):
    if file_name == '__manual entry__':
        return "", ""

    file_path = Path(INPUT_FOLDER) / file_name
    mode_name = file_name.removesuffix('.txt')

    if not file_path.is_file():
        return "It doesn't seem like that file exists anymore.", mode_name

    with open(file_path, 'r') as f:
        file_text = "".join([f"{line.strip()}\n" for line in f.readlines()])

    if len(file_text) > 0:
        return file_text, mode_name

    return "File was there, but it had no text.", mode_name


@app.callback(
    Output('parse-button', 'disabled'),
    Input('mode-name', 'value'),
    Input('raw-text', 'value')
)
def activate_parse_button(name: str, text: str):
    if len(name.strip()) > 0 and len(text.strip()) > 0:
        return False

    return True


@app.callback(
    Output('reset-message-div', 'children'),
    Input('reset-button', 'n_clicks'),
    State('mode-name', 'value'),
)
def reset_mode(nclicks, name):
    if ctx.triggered_id == "reset-button":
        model_path = Path(f'./models/{str(name).strip()}/')

        if model_path.is_dir():

            stopwords_file = model_path / 'stopwords.pickle'
            theoretical_codes_file = model_path / 'theoretical_codes.pickle'

            if stopwords_file.is_file(): stopwords_file.unlink()

            if theoretical_codes_file.is_file(): theoretical_codes_file.unlink()

            for key in assigned_codes:
                assigned_codes[key] = [False] * len(theoretical_code_list)

            return "Existing mode files were cleared. Page refresh is recommended."

        else:
            return "No action taken because existing model couldn't be found."


@app.callback(
    Output('utterances-div', 'children'),
    Output('input-accordion', 'active_item'),
    Output('graph-button', 'disabled'),
    Input('parse-button', 'n_clicks'),
    State('mode-name', 'value'),
    State('raw-text', 'value'),
    State('inclusion-options', 'value'),
    prevent_initial_call=True
)
def utterance_table(parse_clicks, name, txt, options):

    global assigned_codes
    global nlp
    global stopped_words
    global unstopped_words
    global stored_data

    if parse_clicks is not None:

        if parse_clicks > 0:

            model_path = Path(f'./models/{str(name).strip()}/')
            default_stopwords_file = Path('./config') / 'default_stopwords.pickle'

            if not model_path.is_dir():
                with open(default_stopwords_file, 'rb') as dswf:
                    stopped_words = pickle.load(dswf)
            else:
                stopwords_file = model_path / 'stopwords.pickle'
                theoretical_codes_file = model_path / 'theoretical_codes.pickle'

                if stopwords_file.is_file():
                    with open(stopwords_file, 'rb') as swf:
                        stopped_words, unstopped_words = pickle.load(swf)
                else:
                    with open(default_stopwords_file, 'rb') as dswf:
                        stopped_words = pickle.load(dswf)

                if theoretical_codes_file.is_file():
                    with open(theoretical_codes_file, 'rb') as tcf:
                        saved_codes = pickle.load(tcf)

                    assigned_codes = saved_codes

            # reload the model because it only pulls default stopwords when loading
            nlp = spacy.load('en_core_web_sm', exclude=["ner", "senter"])

            nlp(' '.join(stopped_words))
            nlp(' '.join(unstopped_words))

            for word in stopped_words:
                nlp.vocab[word].is_stop = True

            for word in unstopped_words:
                nlp.vocab[word].is_stop = False

            time = True if 0 in options else False
            speaker = True if 1 in options else False
            interviewer = False if 2 in options else True

            parsed_data = parse_raw_text(txt,
                                         timestamp=time,
                                         speaker=speaker,
                                         interviewer=interviewer)

            column_names = [{'name': 'line', 'id': 'line'}]

            if time:
                column_names.append({'name': 'time', 'id': 'time'})

            if speaker:
                column_names.append({'name': 'speaker', 'id': 'speaker'})

            column_names.append({'name': 'utterance', 'id': 'utterance'})

            transcript_table = DataTable(
                parsed_data,
                columns=column_names,
                style_header={
                    'fontWeight': 'bold',
                    'textAlign': 'left'
                },
                style_cell={
                    'padding': '12px',
                    'textAlign': 'left',
                    'fontSize': 16,
                    'line-height': '2',
                    'font-family': 'sans-serif'
                },
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 252, 253)'
                    }
                ],
                id='data-table'
            )

            editor_section = [
                transcript_table
            ]

            stored_data = parsed_data

            return editor_section, "1", False
        else:
            message = [html.P('Processed text will be displayed here as a datatable.', className='lead')]
            return message, "0", True


@app.callback(
    Output('token-buttons', 'children'),
    Output('utterance-stats', 'children'),
    Output('code-checkboxes-container', 'children'),
    Output('coding-modal', 'is_open'),
    Input('data-table', 'active_cell'),
    Input({'type': 'toggle-token', 'index': ALL, 'stop': ALL}, 'n_clicks'),
    Input({'type': 'code-checkbox', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def coding_editor(cell, toggle_clicks, checked_codes):

    global stored_data

    if cell is not None:

        if len(toggle_clicks) > 0:
            if 1 in toggle_clicks:

                toggled_token = ctx.triggered_id['index']
                was_stop = ctx.triggered_id['stop']

                if was_stop:
                    nlp.vocab[toggled_token].is_stop = False
                    stopped_words.discard(toggled_token)
                    unstopped_words.add(toggled_token)
                else:
                    nlp.vocab[toggled_token].is_stop = True
                    stopped_words.add(toggled_token)
                    unstopped_words.discard(toggled_token)

        i, j = cell['row'], 'utterance'
        cell_text = str(stored_data[i][j])
        token_buttons, token_treemap = process_utterance(cell_text)

        line_num = int(stored_data[i]['line'] - 1)
        if len(checked_codes) > 0:
            if type(ctx.triggered_id) is not str:
                if ctx.triggered_id['type'] == 'code-checkbox':
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
    Output('graph-div', 'children'),
    Output('graph-slider', 'max'),
    Output('graph-slider', 'value'),
    Output('metrics-div', 'children'),
    Output('min-co', 'value'),
    Input('graph-button', 'n_clicks'),
    Input('graph-slider', 'value'),
    Input('include-codes', 'value'),
    Input('dmc-mode', 'value'),
    Input('dmc-window', 'value'),
    Input('min-co', 'value'),
    Input('min-dmc-co', 'value'),
    Input('layout-iterations', 'value'),
    Input('node-size', 'value'),
    State('mode-name', 'value'),
    prevent_initial_call=True
)
def knowledge_graph(n_clicks,
                    line,
                    code_pref,
                    dmc,
                    window,
                    mco,
                    mdmco,
                    layout,
                    multiplier,
                    name):

    global stored_data

    # first, let's pickle the user generated model
    pickle_model(name)

    # make sure min co-occurrence is not larger than min dmc co-occurrence
    mco = mco if mco < mdmco else mdmco

    # display the latest utterance when generating a cumulative layout
    if not dmc:
        line = line if line != 1 else len(stored_data)

    start = 0
    end = line

    end = end if end < len(stored_data) else len(stored_data)

    if dmc:
        r = int((window - 1) / 2)
        start = max(0, line - r)
        end = min(len(stored_data), line + r)

        if window == 1:
            num_lines = start + 1

    graph, stats = generate_graph(start_line=start,
                                  end_line=end,
                                  case_name=name,
                                  with_codes=code_pref,
                                  layout_iterations=layout,
                                  min_co=mco,
                                  min_dmc_co=mdmco,
                                  node_size_multiplier=multiplier
                                  )

    return graph, len(stored_data), line, stats, mco


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run_server(debug=True)
