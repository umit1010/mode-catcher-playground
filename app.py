import re
import dash
import dash_bootstrap_components as dbc
import spacy
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from dash.dash_table import DataTable
from dash import Dash, dcc, callback, html, Input, Output, State

# ---- NLP ----


nlp = spacy.load('en_core_web_lg')


def parse_raw_text(txt: str,
                   include_timestamp=False,
                   include_speaker=False,
                   include_interviewer=False):
    input_lines = [line.strip() for line in txt.splitlines()]

    re_time_splitter = re.compile(r'(\[[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\])')

    data = []

    for i, line in enumerate(input_lines):
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
    df = pd.DataFrame.from_dict(token_counts, orient='index') \
        .reset_index() \
        .rename(columns={'index': 'token', 0: 'count'}) \
        .sort_values(by=['count'], ascending=False)

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


# ---- INTERFACE ----


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

with open('sample-short.txt', 'r') as f:
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

utterances_div = html.Div(id='utterances-div',
                          className='border rounded p-4')

theoretical_codes_list = [
    'emergence',
    'thinking in levels',
    'stochasticity',
    'feedback'
]

code_table_rows = [
    html.Tbody([
        html.Tr([
            html.Td(
                dbc.Switch(
                    label=code,
                    value=False,
                    className='p-1'
                )
            ),
            html.Td(
                dbc.RadioItems(
                    options=[
                        {"label": "yes", "value": -1},
                        {"label": "vague", "value": 0},
                        {"label": "no", "value": 1},
                    ],
                    value=0,
                    inline=True,
                    id=f'code-option-{code}',
                    class_name='small'
                )
            )
        ])
        for code in theoretical_codes_list
    ])
]

coding_modal = dbc.Modal(
    [
        dbc.ModalHeader(
            dbc.ModalTitle('Modify'),
            close_button=True
        ),
        dbc.ModalBody(
            dbc.Row(
                [
                    dbc.Col('', id='token-buttons'),
                    dbc.Col(
                        [
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
                                        html.P('not yet functional', className='lead'),
                                        dbc.Table(
                                            code_table_rows,
                                            class_name='border',
                                            bordered=True,
                                            hover=True,
                                            responsive=True,
                                            striped=True
                                        )
                                    ]
                                ), class_name='mt-4'
                            )
                        ]
                    )
                ]
            )
        )
    ],
    id='coding-modal',
    # fullscreen=True,
    scrollable=True,
    size='xl',
    is_open=False,
    centered=True
)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H1(children='mode-catcher playground', className='text-center m-4'),
                    input_accordion,
                    html.P('')
                ]
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        utterances_div,
                        html.P('')
                    ]
                )
            ]
        ),
        coding_modal
    ],
    fluid=True,
    class_name='p-4'
)


# ---- CALLBACKS ----


@app.callback(
    Output('utterances-div', 'children'),
    Output('input-accordion', 'active_item'),
    Input('parse-button', 'n_clicks'),
    State('raw-text', 'value'),
    State('inclusion-options', 'value')
)
def do(c, txt, options):
    if c is not None:
        if c > 0:

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
                    'textAlign': 'center'
                },
                style_cell={
                    'padding': '16px',
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
                html.H3('Utterance Editor', className='my-4'),
                html.P('Click a row to edit.', className='lead my-4'),
                transcript_table
            ]

            return editor_section, "1"

        else:
            return "", "0"


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run_server(debug=True)
