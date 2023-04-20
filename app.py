import re
import dash
import dash_bootstrap_components as dbc
import spacy
import pandas as pd
from dash.dash_table import DataTable
from dash import Dash, html, callback, Input, Output, State

nlp = spacy.load('en_core_web_lg')


### NLP FUNCTIONS

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


### DASH APP INTERFACE

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
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
    id='inclusion-options'
)

input_accordion = dbc.Accordion(
    [
        dbc.AccordionItem(
            [
                raw_input,
                html.P(''),
                dbc.Row([
                    dbc.Col(parse_button),
                    dbc.Col(inclusion_options)
                ]),
                html.P(''),
            ],
            title="Input",
            item_id='0')
     ],
    active_item=0,
    id="input-accordion"
)

utterances_container = dbc.Container(id='utterances-container', class_name='border rounded p-2')

app.layout = dbc.Container([
    html.H1(children='demo playground', className='text-center m-4'),
    input_accordion,
    html.P(''),
    utterances_container,
    html.P('', className='m-4'),
])


@app.callback(
    Output('utterances-container', 'children'),
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
                style_data_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 252, 253)'
                }]
            )

            editor_section = [
                html.H3('Utterance Editor', className='mx-2 mt-2 mb-4'),
                transcript_table
            ]

            return editor_section, "1"

        else:
            return "", "0"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run_server(debug=True)
