import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

# Настройка страницы
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# Навигация
pages = {
    "Анализ и модель": analysis_and_model_page,
    "Презентация": presentation_page
}

# Боковое меню
st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти к", list(pages.keys()))

# Отображение выбранной страницы
pages[selection]()