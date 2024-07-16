import pickle
from pathlib import Path

import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import streamlit_authenticator as stauth  # pip install streamlit-authenticator

names = ["Avidity Associate"]
usernames = ["Avidity"]
passwords = ["Avidity2014!"]

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(r'C:\Users\Admin\Desktop\Potency_app\generate_keys.py').parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)