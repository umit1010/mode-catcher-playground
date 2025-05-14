from datetime import datetime
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/aggregates", title=f"aggregate data | mode-catcher | dev build ({datetime.today()})")

layout = html.Div([
    html.H1(['mode-catcher ', html.Em('aggregate statistics')]),
])

layout = dbc.Container(
    [
        html.H1(['mode-catcher ', html.Em('aggregate statistics')]),
    ],
    fluid=True,
    class_name="p-4",
)
