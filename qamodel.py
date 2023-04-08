import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


pipe = pickle.load(open('pipepipe.pkl', 'rb'))
pipe2 = pickle.load(open('pipepipe2.pkl', 'rb'))

st.title("Question-Answering Model.")
context = st.text_input('Context')
question = st.text_input('Question')

if st.button('Submit'):
    input_data = {'question': question, 'context': context}
    result = pipe.predict(input_data)
    result2 = pipe2.predict(input_data)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('RoBERTa')
        st.write(f"Answer : {result['answer']}")
        st.write(f"Score : {result['score']}")
    with col2:
        st.subheader('BERT')
        st.write(f"Answer : {result2['answer']}")
        st.write(f"Score : {result2['score']}")
