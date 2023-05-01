import dash
import dash_bootstrap_components as dbc
import datetime
import networkx as nx
import numpy as np
import re
import spacy
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from dash.dash_table import DataTable
from dash import Dash, ALL, ctx, dcc, callback, html, Input, Output, State
from pathlib import Path

# ---- PLATFORM ----

nlp = spacy.load('en_core_web_sm', exclude=["ner", "senter"])

stopped_words = set()
unstopped_words = set()

assigned_codes = dict()

theoretical_code_list = [
    'emergent',
    'centralized',
    'probabilistic',
    'deterministic',
    'deliberate feedback loop',
    'pattern fitting',
    'thinking in levels',
    'level slippage',
]

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# ---- NLP ----


def parse_raw_text(txt: str,
                   include_timestamp=False,
                   include_speaker=False,
                   include_interviewer=False):
    input_lines = [line.strip() for line in txt.splitlines()]

    re_time_splitter = re.compile(r'(\[[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\])')

    data = []

    for i, line in enumerate(input_lines):

        if len(line.replace('\n', '').strip()) == 0:
            continue

        _, time, speaker_speech = re_time_splitter.split(line)
        speaker, utterance = speaker_speech.strip().split(':')
        speaker = str(speaker)

        if not include_interviewer and speaker.lower().find('interviewer') > -1:
            continue

        row = {'line': i}

        if include_timestamp:
            row['time'] = time

        if include_speaker:
            row['speaker'] = speaker

        row['utterance'] = utterance.strip()

        data.append(row)

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
    doc = nlp(raw_text)

    all_tokens = [token.lemma_ for token in doc if not nlp.vocab[token.lemma].is_stop
                  and not nlp.vocab[token.lemma].is_punct]
    token_counts = Counter(all_tokens)
    data_dict = {'token': list(token_counts.keys()), 'count': list(token_counts.values())}
    df = pd.DataFrame.from_dict(data_dict)

    # buttons_for_text = None

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

    models_folder = Path('./models/')
    models_folder.mkdir(exist_ok=True)

    mode_folder = models_folder / mode_name
    mode_folder.mkdir(exist_ok=True)

    stopwords_file = mode_folder / 'stopwords.pickle'
    theoretical_codes_file = mode_folder / 'theoretical_codes.pickle'

    with open(stopwords_file, 'wb') as swf:
        pickle.dump({'stopped': stopped_words,
                     'unstopped': unstopped_words},
                    swf,
                    protocol=pickle.HIGHEST_PROTOCOL)

    with open(theoretical_codes_file, 'wb') as tcf:
        pickle.dump(assigned_codes, tcf, protocol=pickle.HIGHEST_PROTOCOL)


def generate_co_occurrence_graph(data_dict_list, model=None):
    # generate unique lemmas list and create a co-occurrence matrix dataframe
    combined_text = " ".join([line['utterance'] for line in data_dict_list])
    combined_doc = model(combined_text)
    all_tokens = [token.lemma for token in combined_doc if
                  not nlp.vocab[token.lemma].is_punct and not nlp.vocab[token.lemma].is_stop]
    token_counts = Counter(all_tokens)
    unique_tokens = list(token_counts.keys())
    df = pd.DataFrame({'token': unique_tokens}, columns=unique_tokens, index=unique_tokens).fillna(0)

    # first, fill each token's counts in the matrix ([same col, same row] = count)
    for token in unique_tokens:
        df.loc[token, token] = token_counts[token]

    # next, iterate over each line's unique tokens and add them to the matrix
    for line in data_dict_list:
        line_doc = nlp(line['utterance'])
        line_tokens = list(set([token.lemma for token in line_doc if
                                not nlp.vocab[token.lemma].is_punct and not nlp.vocab[token.lemma].is_stop]))

        # first loop is iterating over each token
        # second loop iterates over the tokens after the current token
        for i in range(len(line_tokens)):
            row = line_tokens[i]
            for j in range(i + 1, len(line_tokens)):
                col = line_tokens[j]
                df[row][col] += 1
                df[col][row] += 1

    # lastly, create the graph
    labels = [nlp.vocab.strings[token] for token in unique_tokens]

    # node_labels = dict([(token, nlp.vocab.strings[token]) for token in unique_tokens])
    node_sizes = [df[token][token] * 5 for token in unique_tokens]

    nodes = [(token, {'weight': df[token][token]}) for token in unique_tokens]

    edges = []

    for i in range(len(unique_tokens)):
        row = unique_tokens[i]
        for j in range(i + 1, len(unique_tokens)):
            col = unique_tokens[j]
            if df[row][col] > 1:
                edges.append((row, col))

    # create the network
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # generate a spring layout for node locations
    layout_seed = np.random.RandomState(42)
    pos = nx.spring_layout(G, iterations=3, seed=layout_seed)

    edge_x = []
    edge_y = []

    for edge in G.edges():
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
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # plot the nodes

    # node_x = []
    # node_y = []

    # for n in pos:
    #     x, y = pos[n]
    #     node_x.append(x)
    #     node_y.append(y)

    node_x = [pos[i].tolist()[0] for i in pos]
    node_y = [pos[i].tolist()[1] for i in pos]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hovertext=labels,
        hoverinfo='text',
        text=labels,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='Portland',
            reversescale=False,
            color=node_sizes,
            size=node_sizes,
            line_width=1
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            hovermode='closest',
            height=800,
            showlegend=False
        )
    )

    fig.update_xaxes(range=[-1.5, 1.5])
    fig.update_yaxes(range=[-1.5, 1.5])

    return dcc.Graph(figure=fig)


# ---- INTERFACE ----

stored_data = dcc.Store(id='stored-data', storage_type='memory')

# -- input section --

INPUT_FOLDER = 'samples'

input_folder_path = Path(INPUT_FOLDER)

file_list = ['__manual entry__']

if input_folder_path.is_dir():
    text_files = [f.name for f in input_folder_path.glob('*.txt')]
    if len(text_files) > 0:
        file_list.extend(text_files)

input_file_dropdown = dbc.Select(
    file_list,
    id='input-file-dropdown',
    value='demo.txt'
)

INITIAL_INPUT = 'demo'

with open(f'{INPUT_FOLDER}/{INITIAL_INPUT}.txt', 'r') as f:
    sample_text = "".join([f"{line.strip()}\n" for line in f.readlines()])

mode_name = dbc.Input(id='mode-name',
                            value=INITIAL_INPUT,
                            placeholder="Enter mode name ...")

raw_text = dbc.Textarea(
    placeholder="Copy and paste some text here.",
    value=sample_text,
    rows=10,
    id='raw-text'
)

parse_button = dbc.Button('Parse Utterances',
                          id='parse-button',
                          n_clicks=0)

inclusion_options = dbc.Checklist(
    options=[
        {'label': 'Display Timestamp', 'value': 0},
        {'label': 'Display Speaker', 'value': 1},
        {'label': 'Ignore Interviewer Utterances', 'value': 2},
        {'label': 'Reset Model', 'value':3}
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
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Input File:'),
                        input_file_dropdown,
                        html.P('')
                    ], width=10, lg=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Mode name:'),
                        mode_name,
                        html.P('')
                    ], width=10, lg=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Transcript:'),
                        raw_text,
                        html.P('')
                    ])
                ]),
                dbc.Row(
                    dbc.Col(
                        [
                            inclusion_options,
                            parse_button
                        ]
                    ),
                ),
                html.P(''),
            ],
            title="Input",
            item_id='0')
    ],
    active_item='0',
    id="input-accordion"
)

# -- utterances section --

graph_options = dbc.Checklist(
    options=[
        {'label': 'Include Theoretical Codes', 'value': 0},
        {'label': 'DMC Mode', 'value': 1},
        {'label': 'Save Model', 'value': 2}
    ],
    value=[2],
    inline=True,
    class_name='mt-4',
    id='graph-options'
)

generate_graph = dbc.Button('Generate Graph',
                            id='graph-button',
                            class_name='mt-4',
                            n_clicks=0,
                            disabled=True)

utterances_wrapper_div = html.Div(
    [
        html.H3('Utterance Editor', className='mb-4'),
        html.Div(
            [
                html.P('Processed text will be displayed here as a datatable.', className='lead')
            ], id='utterances-div'
        ),
        html.P(' '),
        graph_options,
        generate_graph
    ], className='border rounded p-4'
)

theoretical_codes_list = [
    'emergent',
    'centralized',
    'probabilistic',
    'deterministic',
    'deliberate feedback loop',
    'pattern fitting',
    'thinking in levels',
    'level slippage',
]

code_checkboxes_container = dbc.Container(
    "",
    fluid=True,
    class_name='d-flex align-content-start flex-wrap',
    id='code-checkboxes-container'
)

# -- graph view --

graph_view_wrapper_div = html.Div(
    [
        html.H3('Knowledge Graph', className='mb-4'),
        html.Div(
            [
                html.P(
                    'Knowledge graph will be displayed once you generate it.',
                    className='lead'
                )
            ], id='graph-div'
        ),
        dcc.Slider(min=1, max=2, step=1, value=1, id='graph-slider')
    ], className='border rounded p-4'
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
                html.P('')
            ])
        ),
        dbc.Row([
            dbc.Col([
                utterances_wrapper_div,
                html.P('')
            ])
        ]),
        dbc.Row([
            dbc.Col([graph_view_wrapper_div, html.P('')])
        ]),
        stored_data,
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
    Output('utterances-div', 'children'),
    Output('stored-data', 'data'),
    Output('input-accordion', 'active_item'),
    Output('graph-button', 'disabled'),
    Input('parse-button', 'n_clicks'),
    State('mode-name', 'value'),
    State('raw-text', 'value'),
    State('inclusion-options', 'value'),
    prevent_initial_call=True
)
def utterance_table(parse_clicks, mode_name, txt, options):

    global stopped_words
    global unstopped_words
    global assigned_codes

    if parse_clicks is not None:

        if parse_clicks > 0:

            # # if "Reset Vocabulary" option is not selected
            # # and if there's already an existing vocabulary, load it from the disk
            # # otherwise, create a new blank vocabulary
            model_path = Path(f'./models/{str(mode_name).strip()}/')
            if model_path.is_dir():
                stopwords_file = model_path / 'stopwords.pickle'
                theoretical_codes_file = model_path / 'theoretical_codes.pickle'

                if stopwords_file.is_file():

                    with open(stopwords_file, 'rb') as swf:
                        loaded_stopwords = pickle.load(swf)

                    stopped_words = loaded_stopwords['stopped']
                    unstopped_words = loaded_stopwords['unstopped']

                    nlp.Defaults.stop_words.update(stopped_words)
                    nlp.Defaults.stop_words.discard(unstopped_words)

                if theoretical_codes_file.is_file():

                    with open(theoretical_codes_file, 'rb') as tcf:
                        saved_codes = pickle.load(tcf)

                    assigned_codes = saved_codes

            time = True if 0 in options else False
            speaker = True if 1 in options else False
            interviewer = False if 2 in options else True

            parsed_data = parse_raw_text(txt,
                                         include_timestamp=time,
                                         include_speaker=speaker,
                                         include_interviewer=interviewer)

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

            return editor_section, parsed_data, "1", False
        else:
            message = [html.P('Processed text will be displayed here as a datatable.', className='lead')]
            return message, "Nothing is parsed yet!", "0", True


@app.callback(
    Output('token-buttons', 'children'),
    Output('utterance-stats', 'children'),
    Output('code-checkboxes-container', 'children'),
    Output('coding-modal', 'is_open'),
    Input('data-table', 'active_cell'),
    Input({'type': 'toggle-token', 'index': ALL, 'stop': ALL}, 'n_clicks'),
    Input({'type': 'code-checkbox', 'index': ALL}, 'value'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def coding_editor(cell, toggle_clicks, checked_codes, data):

    if cell is not None:

        if len(toggle_clicks) > 0:
            if 1 in toggle_clicks:

                toggled_token = ctx.triggered_id['index']
                was_stop = ctx.triggered_id['stop']

                if was_stop:
                    nlp.vocab[toggled_token].is_stop = False
                    unstopped_words.add(toggled_token)
                    stopped_words.discard(toggled_token)
                else:
                    nlp.vocab[toggled_token].is_stop = True
                    unstopped_words.discard(toggled_token)
                    stopped_words.add(toggled_token)

        i, j = cell['row'], 'utterance'
        cell_text = str(data[i][j])
        token_buttons, token_treemap = process_utterance(cell_text)

        line_num = int(data[i]['line'])
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
    Input('graph-button', 'n_clicks'),
    Input('graph-slider', 'value'),
    State('stored-data', 'data'),
    State('mode-name', 'value'),
    State('graph-options', 'value'),
    prevent_initial_call=True
)
def knowledge_graph(n_clicks, slider_value, data, name, options):
    if n_clicks is not None:

        if n_clicks > 0 and ctx.triggered_id == 'graph-button':

            if 0 in options:
                print("You want me to include theoretical codes")

            if 1 in options:
                print("You want me to show the DMC viewer, not the cumulative one")

            if 2 in options:
                pickle_model(name)

            return generate_co_occurrence_graph(data[0:1], model=nlp), len(data), 0

        elif n_clicks > 0 and slider_value > 1 and ctx.triggered_id == 'graph-slider':
            return generate_co_occurrence_graph(data[0:slider_value], model=nlp), len(data), slider_value

    else:

        message = [html.P('Knowledge graph will be displayed here once utterances are processed.')]
        return message, 2, 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run_server(debug=True)
