import streamlit as st
from modeling_dl import serve_model_dl
from modeling_non_dl import serve_model_non_dl


def process_sentence(input_sentence):
    dl = serve_model_dl(model_path='models/model-dl.pth', vocab_path='models/vocab.pkl', sentence=input_sentence)
    nondl = serve_model_non_dl(model_path='models/model-lr.pkl', vec_path='models/vec.pkl', sentence=input_sentence)
    return dl, nondl


def main():
    st.title("Sentence Processor")
    input_sentence = st.text_input("Enter a sentence:")

    if st.button("Predict"):
        if input_sentence:
            dl, nondl = process_sentence(input_sentence)
            st.write(f'results from deep learning model: {dl}')
            st.write(f'results from non deep learning model: {nondl}')


if __name__ == "__main__":
    main()
