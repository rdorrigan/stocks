from dash import Dash
import os
import argparse
import warnings
from layout.layout import create_layout, get_style_sheets
from data.db import initial_setup
warnings.filterwarnings('ignore', category=FutureWarning)
PROD = True
DEPRECATED = False

# Initialize the app
dash_app = Dash(__name__, external_stylesheets=get_style_sheets())
# server = dash_app.server  # Needed for deployment

# Ensure a valid port is assigned
PORT = int(os.environ.get("PORT", 8050))

# Fetch top stocks dynamically

initial_setup()

# Layout
dash_app.layout = create_layout()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prod', action='store_false')
    parser.add_argument('-dep', '--deprecated', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    return parser.parse_args()


# Run the apps
if __name__ == "__main__":
    args = parse_args()
    if args.prod:
        PROD = True
    else:
        PROD = False
    if args.deprecated:
        DEPRECATED = True
    try:
        # if not PROD:
        if args.debug:
            dash_app.run(debug=True)  # , host="0.0.0.0", port=PORT
        else:
            from waitress import serve
            serve(dash_app.server, host="0.0.0.0", port=PORT)
    except Exception as e:
        print(f"Error starting server: {e}")
