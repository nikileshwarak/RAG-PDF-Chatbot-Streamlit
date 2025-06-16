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


![image](https://github.com/user-attachments/assets/9278e006-ba9b-44a3-94d8-56fd2d146541)

![image](https://github.com/user-attachments/assets/ae6bc4c1-641c-4f97-83fe-56675a2b38a7)

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
