# 🤖 AI RAG Document Processing and Retrieval Example 🚀


This project demonstrates how to use LangChain and its community-supported components for advanced document processing, question-answering, and vector-based retrieval. Whether you're working with PDFs, HTML pages, or embeddings, this example showcases how to integrate LangChain's functionality into your projects.

---

## 🎯 Features

- **📄 PDF Handling**: Load and split PDF documents for text processing.
- **🔍 Vector Search**: Efficiently retrieve context using `DocArrayInMemorySearch`.
.
- **🌐 HTML Processing**: Convert and process web pages using `Html2TextTransformer`.

---

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/hosseingz/LangChain-DocProcessor
   cd LangChain-DocProcessor
   ```

2. **Install Dependencies**:
   Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📝 Usage Guide

### 1️⃣ **Load a PDF**

Using `PyPDFLoader`, load and split a PDF document into pages for processing:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Authentication - Django REST framework.pdf")
pages = loader.load_and_split()
```

---

### 2️⃣ **Setup Embeddings and Language Model**

Define the embeddings and LLM (Language Learning Model) using `Ollama`:

```python
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

MODEL = "phi3:mini"  # Replace with your preferred model
embedding = OllamaEmbeddings(model=MODEL)
model = Ollama(model=MODEL)
```

---

### 3️⃣ **Create a Prompt Template**

Define the logic for interacting with the LLM by using `PromptTemplate`:

```python
from langchain.prompts import PromptTemplate

template = """
Answer the question based on the context below. If you can't answer the questions, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
```

---

### 4️⃣ **Build the Retrieval System**

Leverage `DocArrayInMemorySearch` to create a vector-based retrieval system:

```python
from langchain_community.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embedding)
retriever = vectorstore.as_retriever()
```

Now, the retriever can fetch relevant content for your queries.

---

### 5️⃣ **Chain Everything Together**

Connect all the components into a pipeline:

```python
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

parser = StrOutputParser()

chain = (
    {
        'context': itemgetter('question') | retriever,
        'question': itemgetter('question')
    }
    | prompt
    | model
    | parser
)
```

Invoke the chain to answer a question:

```python
res = chain.invoke({'question': 'What is the purpose of this context?'})
print(res)
```

---

### 6️⃣ **Process HTML Documents**

Load HTML content asynchronously and convert it to text:

```python
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader

urls = "https://en.wikipedia.org/wiki/Cyrus_the_Great"
html2text = Html2TextTransformer()
loader = AsyncHtmlLoader(urls)

docs = loader.load()
```

This process extracts meaningful content from web pages.


---

## ⚙️ Dependencies

This project requires the following Python libraries:

- `langchain` and `langchain-community`: Core libraries for building the pipeline.
- `pypdf`: To process PDF documents.
- `docarray`: For vector storage and search.
- `pydantic`: For data validation and model management.

---

## 🛡️ License

This project is open-source and available under the **MIT License**.

---

## ✨ Acknowledgments

Special thanks to the LangChain community for providing the tools and components used in this project. 🙌

---

Happy coding! 😄