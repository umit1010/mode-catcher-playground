import math
import pickle

from datetime import datetime

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import networkx as nx


dash.register_page(__name__, path="/aggregates", title=f"stats | mode-catcher | dev build ({datetime.today()})")

# ToDo: Load the data from

# ToDo: update the data of this figure through callbacks
fig = px.scatter(x=[5, 9, 3, 11], y=[5, 9, 3, 11])

plotting_variables = [
    {"label": "average degree centrality", "value":"avedeg"},
    {"label": "average clustering coefficient", "value":"aveclus"},
    {"label": "number of nodes", "value":"numnodes"},
    {"label": "number of connected nodes", "value":"numconn"},
    {"label": "mode name", "value":"modename"},
]

layout = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                dbc.Col(
                    dbc.Alert(
                        [
                            html.I(className="bi bi-exclamation-triangle-fill me-2"),
                            "This view is not yet functional."
                        ],
                        color="warning"
                    ),
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("x-axis: "),
                                dbc.Select(options=plotting_variables, value="avedeg"),
                            ],
                            class_name="mx-2",
                        ),
                        xl=4, lg=6, md=12,
                    ),
                    dbc.Col(
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("y-axis: "),
                                dbc.Select(options=plotting_variables, value="avedeg"),
                            ],
                            class_name="mx-2",
                        ),
                        xl=4, lg=6, md=12,
                    )
                ],
                justify="end",
                class_name="mx-1 my-3",
            ),
            dcc.Graph(id='graph', figure=fig, mathjax=True, className="border rounded p-2 m-2"),
            html.Div(
                [
                    html.H5("Included Graphs"),
                    dbc.Checklist(
                        options=[
                            {"label": "aiden1.sm", "value": 0},
                            {"label": "aiden1.lg", "value": 1},
                            {"label": "blaise1.sm", "value": 2},
                            {"label": "blaise1.lg", "value": 3},
                            {"label": "lena1.lg", "value": 4},
                            {"label": "spur1.lg", "value": 5},
                            {"label": "spur2.lg", "value": 6},
                        ],
                        value=[1, 3, 4, 5, 6],
                        inline=True,
                        class_name="ms-1 mt-3",
                    ),
                ],
                className="m-2 p-3 border rounded",
            ),
            html.Div(
                [
                    html.H5("Visualization"),
                    dbc.Stack(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Plot type: "),
                                    dbc.Select(
                                        options=[
                                            {"label": "Scatter", "value": "scat"},
                                            {"label": "Line", "value": "line"},
                                            {"label": "Histogram", "value": "hist"},
                                        ],
                                        value="scat"
                                    ),
                                ]
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Point Size: "),
                                    dbc.Input(type="number", value=1, min=1, max=10),
                                ]
                            ),

                        ],
                        direction="horizontal",
                        gap=4,
                        className="ms-1 mt-3",
                    )
                ],
                className="m-2 p-3 border rounded",
            ),

        ],
    ),
    class_name="m-4"
)

