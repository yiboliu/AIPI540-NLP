import streamlit as st


def process_sentence(input_sentence):
    # Your logic to process the input sentence and generate two output sentences here
    # For this example, we'll just create dummy output sentences
    output_sentence1 = f"Output sentence 1: {input_sentence} (Processed)"
    output_sentence2 = f"Output sentence 2: {input_sentence} (Processed)"
    return output_sentence1, output_sentence2


# Streamlit app
def main():
    st.title("Sentence Processor")
    input_sentence = st.text_input("Enter a sentence:")

    if st.button("Process"):
        if input_sentence:
            output_sentence1, output_sentence2 = process_sentence(input_sentence)
            st.write(output_sentence1)
            st.write(output_sentence2)


if __name__ == "__main__":
    main()
