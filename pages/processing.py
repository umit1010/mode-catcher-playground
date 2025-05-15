import math
import pickle

from datetime import datetime

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import networkx as nx


dash.register_page(__name__, path="/pre-processing", title=f"pre-processing | mode-catcher | dev build ({datetime.today()})")

layout = dbc.Card(
    dbc.CardBody(
        [
            dbc.Alert(
                [
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    "This view is not yet functional."
                ],
                color="warning"
            ),
        ]
    ),
    className="m-4",
)
