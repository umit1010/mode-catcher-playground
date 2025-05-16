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
                    # html.Em(
                        html.Span("", id="active-page-name", className="text-success")
                    # )
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
                                " Stats"
                            ],
                            id="aggregate-stats-button",
                            class_name="mx-2 my-2",
                            color="dark",
                            outline=True,
                        ),
                        href="/aggregates",
                    ),
                    dcc.Link(
                        dbc.Button(
                            [
                                html.I(className="bi bi-file-earmark-break me-2"),
                                " Preprocessing"
                            ],
                            id="pre-processing-button",
                            class_name="ms-2 my-2",
                            color="dark",
                            outline=True,
                        ),
                        href="/pre-processing",
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
    Output("aggregate-stats-button", "disabled"),
    Output("pre-processing-button", "disabled"),
    Input("url", "pathname")
)
def display_active_pagename_callback(url):
    if url == "/":
        return "playground", True, False, False
    elif url=="/aggregates":
        return "stats", False, True, False
    elif url=="/pre-processing":
        return "pre-processing", False, False, True
    else:
        return "", True, True

if __name__ == "__main__":
    app.run(debug=True)