import os.path
import sys

from streamlit.web import cli as stcli


def main():
    entry_file = os.path.join(os.path.dirname(__file__), "ui.py")
    sys.argv = [
        "streamlit",
        "run",
        entry_file,
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
