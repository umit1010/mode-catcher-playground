import re
import dash
import dash_bootstrap_components as dbc
import spacy
import pandas as pd
from dash.dash_table import DataTable
from dash import Dash, html, callback, Input, Output, State

nlp = spacy.load('en_core_web_lg')


### NLP FUNCTIONS

def raw_text_to_dictionary(txt: str):
    input_lines = [line.strip() for line in txt.splitlines()]

    re_time_splitter = re.compile(r'(\[[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\])')

    data = []

    for i, line in enumerate(input_lines):
        _, time, speaker_speech = re_time_splitter.split(line)
        speaker, utterance = speaker_speech.strip().split(':')

        data.append({'line':i, 'utterance': utterance.strip()})

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

input_accordion = dbc.Accordion(
    [dbc.AccordionItem([raw_input,
                       html.P(''),
                       parse_button],
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
    State('raw-text', 'value')
)
def do(c, txt):
    if c is not None:
        if c > 0:
            return DataTable(raw_text_to_dictionary(txt),
                             columns=[{'name': 'line', 'id':'line'},
                                      {'name':'utterance', 'id':'utterance'}],
                             style_cell={'textAlign': 'left', 'fontSize':'10px'},
                             style_data={'whiteSpace': 'normal', 'height': 'auto'}
                             ), "1"
        else:
            return "", "0"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run_server(debug=True)
