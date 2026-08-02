import pandas as pd
import streamlit as st


@st.cache_data
def load_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    return df


def get_feature_columns(df, target_col):
    return [c for c in df.columns if c != target_col]