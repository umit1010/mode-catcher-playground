import dash
import dash_bootstrap_components as dbc
import datetime
import networkx as nx
import numpy as np
import re
import spacy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from dash.dash_table import DataTable
from dash import Dash, dcc, callback, html, Input, Output, State


# ---- PLATFORM ----


nlp = spacy.load('en_core_web_sm')

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

    return data


def process_utterance(raw_text):
    doc = nlp(raw_text)

    all_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    token_counts = Counter(all_tokens)
    data_dict = {'token': list(token_counts.keys()), 'count': list(token_counts.values())}
    df = pd.DataFrame.from_dict(data_dict)

    buttons_for_text = [
        dbc.Button(token.text, color='light', class_name='m-1', size='sm') if token.is_stop
        else html.Span(token.text, className='mx-1') if token.is_punct
        else dbc.Button(token.text, color='warning', class_name='m-1', size='sm')
        for token in doc
    ]

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

    token_treemap = dcc.Graph(figure=fig, responsive=True, style={'height': '200px'})

    return buttons_for_text, token_treemap


def generate_co_occurrence_graph(data_dict_list):

    # generate unique lemmas list and create a co-occurrence matrix dataframe
    combined_text = " ".join([line['utterance'] for line in data_dict_list])
    combined_doc = nlp(combined_text)
    all_tokens = [token.lemma for token in combined_doc if not token.is_punct and not token.is_stop]
    token_counts = Counter(all_tokens)
    unique_tokens = list(token_counts.keys())
    df = pd.DataFrame({'token': unique_tokens}, columns=unique_tokens, index=unique_tokens).fillna(0)

    # first, fill each token's counts in the matrix ([same col, same row] = count)
    for token in unique_tokens:
        df.loc[token, token] = token_counts[token]

    # next, iterate over each line's unique tokens and add them to the matrix
    for line in data_dict_list:
        line_doc = nlp(line['utterance'])
        line_tokens = list(set([token.lemma for token in line_doc if not token.is_punct and not token.is_stop]))

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

    # print(nlp.vocab.strings['yeah'])
    # print(df[11852442279192850303])

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

    node_x = []
    node_y = []

    # for n in pos:
    #     x, y = pos[n]
    #     node_x.append(x)
    #     node_y.append(y)

    node_x = [pos[i].tolist()[0] for i in pos]
    node_y = [pos[i].tolist()[1] for i in pos]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hovertext=labels,
        hoverinfo='text',
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

# stored_data = dbc.Input(value='', id='stored-data', disabled=True, class_name='d-none')
stored_data = dcc.Store(id='stored-data', storage_type='memory')

# -- input section --

with open('sample-medium.txt', 'r') as f:
    sample_text = "".join([f"{line.strip()}\n" for line in f.readlines()])

raw_input = dbc.Textarea(
    placeholder="Copy and paste some text here.",
    value=sample_text,
    rows=10,
    id='raw-text'
)

parse_button = dbc.Button('Parse Utterances', id='parse-button', n_clicks=0)

inclusion_options = dbc.Checklist(
    options=[
        {'label': 'Include Timestamp', 'value': 0},
        {'label': 'Include Speaker', 'value': 1},
        {'label': 'Ignore Interviewer Utterances', 'value': 2},
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
                raw_input,
                html.P(''),
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
    active_item=0,
    id="input-accordion"
)

# -- utterances section --


utterances_wrapper_div = html.Div(
    [
        html.H3('Utterance Editor', className='mb-4'),
        html.Div(
            [
                html.P('Processed text will be displayed here as a datatable.', className='lead')
            ], id='utterances-div'
        ),
        html.P(' '),
        dbc.Button('Generate Graph', id='graph-button', class_name='mt-4', n_clicks=0, disabled=True)
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

code_checkboxes = dbc.Container(
    [
        html.Div([
            dbc.Checkbox(label=code)
        ], className='w-50') for code in theoretical_codes_list
    ],
    fluid=True,
    class_name='d-flex align-content-start flex-wrap'
)

# -- graph view --

graph_view_wrapper_div = html.Div(
    [
        html.H3('Knowledge Graph', className='mb-4'),
        html.Div(
            [
                html.P(
                    'Knowledge graph will be displayed here once utterances are processed.',
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
                                code_checkboxes
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
                html.H1(children='mode-catcher playground', className='text-center m-4'),
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
    Output('utterances-div', 'children'),
    Output('stored-data', 'data'),
    Output('input-accordion', 'active_item'),
    Output('graph-button', 'disabled'),
    Input('parse-button', 'n_clicks'),
    State('raw-text', 'value'),
    State('inclusion-options', 'value')
)
def create_utterance_table(n_clicks, txt, options):
    if n_clicks is not None:
        if n_clicks > 0:

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
                page_size=6,
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
    Output('coding-modal', 'is_open'),
    Input('data-table', 'active_cell'),
    Input('data-table', 'data')
)
def open_coding_editor(cell, data):
    if cell is not None:
        i, j = cell['row'], 'utterance'
        cell_text = str(data[i][j])
        token_buttons, token_treemap = process_utterance(cell_text)

        return token_buttons, token_treemap, True
    else:
        return "Something went wrong!", "Something went wrong!", False


@app.callback(
    Output('graph-div', 'children'),
    Output('graph-slider', 'max'),
    Output('graph-slider', 'value'),
    Input('graph-button', 'n_clicks'),
    Input('stored-data', 'data'),
    Input('graph-slider', 'value')
)
def display_network_graph(n_clicks, data, slider_value):

    if n_clicks is not None:
        if n_clicks > 0 and slider_value == 1:
            return generate_co_occurrence_graph(data[0:1]), len(data), 0
        elif n_clicks > 0 and slider_value > 1:
            return generate_co_occurrence_graph(data[0:slider_value]), len(data), slider_value
    else:
        message = [html.P('Knowledge graph will be displayed here once utterances are processed.')]
        return message, 2, 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run_server(debug=True)
