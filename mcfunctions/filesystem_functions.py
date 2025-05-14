from pathlib import Path

# constants
SAMPLES_FOLDER = Path(__file__).parent.parent / "samples"
MODELS_FOLDER = Path(__file__).parent.parent / "models"
CONFIG_FOLDER = Path(__file__).parent.parent / "config"

DEDUCTIVE_LABEL_DEFINITIONS_FILENAME = "deductive_label_definitions.toml"
DEFAULT_STOPWORDS_FILENAME = "default_stopwords.pickle"

PARSED_DATA_FILENAME = "parsed_data.pickle"
STOPWORDS_FILENAME = "stopwords.pickle"
EXCLUDED_TOKENS_FILENAME = "excluded_tokens.pickle"
EXCLUDED_ROWS_FILENAME = "excluded_rows.pickle"
ASSIGNED_CODES_FILENAME = "assigned_deductive_codes.pickle"
USER_ACTIONS_FILENAME = "user_actions.pickle"

def get_model_path(mode_name, spacy_model, is_sentencized):
    ## create the models folder if it doesn't exist already
    ##     exists_ok = True -> don't overwrite if it already exists
    if not MODELS_FOLDER.is_dir():
        MODELS_FOLDER.mkdir(exist_ok=True)

    ## define the path of the mode (case, participant) for the selected transcript
    mode_path = MODELS_FOLDER / str(mode_name).strip()
    if not mode_path.exists():
        mode_path.mkdir(exist_ok=True)

    ## define the subfolder that corresponds to the specific parsing parameters of the model
    model_path = mode_path / f"{spacy_model}.{'sentencized' if is_sentencized else ''}/"
    if not model_path.exists():
        model_path.mkdir(exist_ok=True)

    return model_path