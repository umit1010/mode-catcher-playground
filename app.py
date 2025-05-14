from datetime import datetime

import dash
import dash_bootstrap_components as dbc
from dash import Dash, ALL, callback, ctx, dcc, html, Input, Output, State


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    use_pages=True,
)
app.title = f"mode-catcher | dev build ({datetime.today()})"

# needed to be able to publish the script on Heroku
server = app.server

app.layout = dbc.Container(
    [
        dcc.Location(id='url', refresh=False),
        dbc.Stack(
            [
                html.H2([
                    "mode-catcher ",
                    html.Em(
                        html.Sub("", id="active-page-name", className="text-primary")
                    )
                ]),
                html.Div([
                    dcc.Link(
                        dbc.Button(
                            [
                                html.I(className="bi bi-virus2 me-2"),
                                " Playground"
                            ],
                            id="playground-button",
                            class_name="m-2",
                            color="primary",
                            outline=True,
                        ),
                        href="/",
                    ),
                    dcc.Link(
                        dbc.Button(
                            [
                                html.I(className="bi bi-bar-chart-line me-2"),
                                " Aggregate Data"
                            ],
                            id="aggregate-data-button",
                            class_name="m-2",
                            color="primary",
                            outline=True,
                        ),
                        href="/aggregates",
                    ),

                ], className="ms-auto")
            ],
            class_name="px-4 pt-3",
            direction="horizontal",
            gap=3,
        ),

        dash.page_container,
    ],
    fluid=True,
    class_name="p-1",
)

@app.callback(
    Output("active-page-name", "children"),
    Output("playground-button", "disabled"),
    Output("aggregate-data-button", "disabled"),
    Input("url", "pathname")
)
def display_active_pagename_callback(url):
    if url == "/":
        return "playground", True, False
    elif url=="/aggregates":
        return "aggregate data", False, True
    else:
        return "", True, True

if __name__ == "__main__":
    app.run(debug=True)