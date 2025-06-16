#import packages
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from datetime import datetime
import base64

import os

#imports for langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# Getting text form pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Getting chunks from text
def get_text_chunks(text,model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Embedding this chunks and storing them in a vector store
def get_vector_store(text_chunks,model_name,api_key=None):
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectore_store = FAISS.from_texts(text_chunks,embedding=embeddings) #Text chunks → converted to vectors
    vectore_store.save_local("faiss_index")
    return vectore_store

# Create a conversational chain using Langchain
def get_conversational_chain(model_name,vectorstore=None,api_key=None):
    if model_name=="Google AI":
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide 
        all the detials with the proper structure, if the answer is not in the provided context just say,
        "Answer is not available in provided context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question:\n {question}?\n

        Answer:
        """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3,google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

# Get the user input - USING STREAMLIT CHAT COMPONENTS
def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("Please upload PDF files and provide API key before processing.")
        return
    
    text_chunks = get_text_chunks(get_pdf_text(pdf_docs), model_name)
    vector_store = get_vector_store(text_chunks, model_name, api_key)
    user_question_output = "" #the user question
    response_output = ""  #the model response
    
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain("Google AI", vectorstore=new_db, api_key=api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        user_question_output = user_question
        response_output = response['output_text']
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

    # Display current conversation using Streamlit chat components
    with st.chat_message("user"):
        st.write(user_question_output)
    
    with st.chat_message("assistant"):
        st.write(response_output)
    
    # Display conversation history
    if len(conversation_history) > 1:
        st.subheader("Previous Conversations:")
        # Show all but the last conversation (which is the current one)
        for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history[:-1]):
            with st.chat_message("user"):
                st.write(question)
                st.caption(f"Asked on: {timestamp}")
            
            with st.chat_message("assistant"):
                st.write(answer)
                st.caption(f"Model: {model_name} | PDF: {pdf_name}")

    # Download conversation
    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the Download button on the left side at the bottom of the conversation.")
    st.snow()  

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    linkedin_profile_link = "https://www.linkedin.com/in/nikileshwarak/"
    kaggle_profile_link = "https://www.kaggle.com/nikileshwar11"
    github_profile_link = "https://github.com/nikileshwarak"

    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
        f"[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle_profile_link}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )

    model_name = st.sidebar.radio("Select the Model:", ( "Google AI"))

    api_key = None

    if model_name == "Google AI":
        api_key = st.sidebar.text_input("Enter your Google API Key:")
        st.sidebar.markdown("Click [here](https://ai.google.dev/) to get an API key.")
        
        if not api_key:
            st.sidebar.warning("Please enter your Google API Key to proceed.")
            return

    with st.sidebar:
        st.title("Menu:")
        
        col1, col2 = st.columns(2)
        
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []  # Clear conversation history
            st.session_state.user_question = None  # Clear user question input 
            
            api_key = None  # Reset Google API key
            pdf_docs = None  # Reset PDF document
            
        else:
            if clear_button:
                if 'user_question' in st.session_state:
                    st.warning("The previous query will be discarded.")
                    st.session_state.user_question = ""  # Temizle
                    if len(st.session_state.conversation_history) > 0:
                        st.session_state.conversation_history.pop()  # Son sorguyu kaldır
                else:
                    st.warning("The question in the input will be queried again.")

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done")
            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""  # Clear user question input 

if __name__ == "__main__":
    main()