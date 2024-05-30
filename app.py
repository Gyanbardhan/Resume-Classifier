
import os
import pandas as pd
import numpy as np
import streamlit as st
import re
import pickle
def remove_tags(text):
    return re.sub(re.compile('<.*?>'),'',text)

def lwr(text):
    return text.lower()
    
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
sw_list=stopwords.words('english')

def stopword(text):
    return " ".join([word for word in text.split() if word not in sw_list])

import string 
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def dec_vector(text):
    with open("tfidf.pkl", 'rb') as file:  
        tfidf = pickle.load(file)
    return tfidf.transform(text)

def preprocessed(text):
    
    text=remove_tags(text)
    text=lwr(text)
    text=stopword(text)
    text=remove_punctuation(text)
    text=[text]
    x=dec_vector(text)
    return x

def clear_text():
    st.session_state["text"] = ""


def main():
    
    
    with open("resumeclassifier.pkl", 'rb') as file1:  
        rf = pickle.load(file1)
    st.title('Resume Classifier')
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf','docx'])
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_bytes = uploaded_file.read()
            text = resume_bytes.decode('latin-1')
    doc={3: 'Frontend Developer',
            0: 'Backend Developer',
            7: 'Python Developer',
            2: 'Data Scientist',
            4: 'Full Stack Developer',
            6: 'Mobile App Developer (iOS/Android)',
            5: 'Machine Learning Engineer',
            1: 'Cloud Engineer'}
    if st.button('Classify'):
        z=preprocessed(text)
        st.success(doc[rf.predict(z)[0]])
        st.button("Clear", on_click=clear_text)


if __name__=='__main__':
    main()
