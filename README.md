# 📚 RAG-Based PDF Chatbot using Streamlit + Gemini + FAISS

Chat with **multiple PDF files** using a conversational **RAG (Retrieval-Augmented Generation)** chatbot built using **LangChain**, **FAISS**, **Google Gemini API**, and deployed with **Streamlit**. Upload documents and ask context-based questions to get accurate, structured answers.

---

##  Features

-  Upload and query multiple PDF files at once
-  Uses Google Gemini 1.5 API for fast, accurate responses
-  Retrieval-Augmented Generation using LangChain and FAISS
-  Clean and interactive chat UI built with Streamlit
-  Download full conversation history as CSV
-  Deployable with Streamlit Cloud

---

##  Screenshots


![image](https://github.com/user-attachments/assets/78eb3344-efe0-4ee7-96cb-d80b672c4a40)

![image](https://github.com/user-attachments/assets/12b07545-83bb-44cc-ac92-625dc0f3e2cc)

![image](https://github.com/user-attachments/assets/25bb4f8e-aa0e-4d6d-ba4f-619e311c19e2)



---

##  Tech Stack

| Layer                  | Technology & Tools                            |
|------------------------|-----------------------------------------------|
| Frontend UI            | `Streamlit`                                   |
| Backend / Logic        | `LangChain`, `Google Gemini API`, `FAISS`     |
| Embeddings             | `Google Generative AI Embeddings`             |
| PDF Parsing            | `PyPDF2`                                      |
| Vector Store           | `FAISS`                                       |
| LLM Model              | `gemini-1.5-flash`                            |
| Prompt Engineering     | `LangChain PromptTemplate`                    |
| Deployment             | `Streamlit Cloud`                             |

---

##  Setup Instructions

1. Clone the Repository:
```
git clone https://github.com/yourusername/RAG-PDF-Chatbot-Streamlit.git
cd RAG-PDF-Chatbot-Streamlit
```
2. Create and Activate Virtual Environment:

```
python -m venv myenv
source myenv/bin/activate     # On Windows: myenv\Scripts\activate
```
3. Install Dependencies:
```
pip install -r requirements.txt
```
4. Get Google API Key:

  Go to https://ai.google.dev/
  Generate your Gemini API key
  Paste it into the sidebar when running the app

5. Run the App:
```
streamlit run app.py
```

##  How to Deploy on Streamlit Cloud

1. Push your code to a public GitHub repository.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. Click **"New app"** → select your repo, branch, and `app.py` file.
4. Click **"Deploy"** — that’s it! Your app will be live.

   
##  Live Demo

Access the deployed app here: [Chat with Multiple PDFs (Demo)](https://rag-pdf-chatbot-app.streamlit.app/)
