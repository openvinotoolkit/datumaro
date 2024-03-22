import os
import sys

from cryptography.fernet import Fernet
from streamlit.web import cli as stcli

if __name__ == "__main__":
    st_server_base_url_key = Fernet.generate_key()
    os.environ["STREAMLIT_SERVER_BASE_URL_PATH"] = st_server_base_url_key.decode("utf-8")

    sys.argv = ["streamlit", "run", "streamlit_app.py"]
    sys.exit(stcli.main())
