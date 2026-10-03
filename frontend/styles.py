import streamlit as st


def apply_styles():

    st.markdown(
        """
        <style>

        .main {
            padding-top: 1rem;
        }

        .neravu-title {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .neravu-subtitle {
            text-align: center;
            font-size: 20px;
            margin-bottom: 25px;
        }

        .language-box {
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 20px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )