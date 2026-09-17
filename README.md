# RAG MVP – Agent Assist

A small RAG-based customer service assistant built to explore how AI can help agents find information from internal documents and respond to customer queries.

The project uses PDF documents as the knowledge base, converts the content into embeddings, stores them in FAISS, and retrieves relevant information when a user asks a question. It also includes a simple intent classification model and conversation memory.

## How it works

```text
PDF Documents
     ↓
Document Ingestion
     ↓
Text Chunking
     ↓
Sentence Transformer Embeddings
     ↓
FAISS Vector Store
     ↓
Retriever
     ↓
RAG Engine + Llama 2.3
     ↓
Agent Assist Application
```

The project also uses **Logistic Regression** for intent classification and keeps track of conversation context.

## Project Structure

| File                          | Description                                            |
| ----------------------------- | ------------------------------------------------------ |
| `01_Document_Ingestion.ipynb` | Loads and processes PDF documents                      |
| `02_Chunking.ipynb`           | Splits documents into smaller text chunks              |
| `03_Embeddings.ipynb`         | Creates embeddings using a Sentence Transformer model  |
| `04_VectorFAISS.ipynb`        | Creates the FAISS vector index                         |
| `05_Retrieval.ipynb`          | Tests document retrieval                               |
| `app.py`                      | Main Streamlit application                             |
| `agent_assist.py`             | Handles the agent-assist functionality                 |
| `rag_engine.py`               | Connects retrieval with the LLM to generate responses  |
| `retriever.py`                | Handles retrieval from the FAISS index                 |
| `llm.py`                      | Handles the Ollama/Llama 2.3 integration               |
| `pipeline.py`                 | Coordinates different parts of the processing pipeline |
| `conversation_memory.py`      | Maintains conversation context                         |
| `intent_tracker.py`           | Tracks customer intent during conversations            |
| `intent_model.pkl`            | Saved Logistic Regression intent model                 |
| `vectorizer.pkl`              | Saved text vectorizer used by the intent model         |
| `requirements.txt`            | Python dependencies                                    |
| `environment.yml`             | Conda environment configuration                        |

## Technologies Used

* **Python**
* **Jupyter Notebooks**
* **Streamlit**
* **FAISS**
* **Sentence Transformers**
* **Ollama**
* **Llama 2.3**
* **Scikit-learn**
* **Logistic Regression**
* **PDF document processing**

## RAG Pipeline

### 1. Document ingestion

PDF documents are loaded and processed to create the knowledge base used by the application.

### 2. Chunking

The extracted text is divided into smaller chunks so that relevant sections can be retrieved without passing entire documents to the LLM.

### 3. Embeddings

The document chunks are converted into numerical vectors using a Sentence Transformer embedding model (`all-mpnet-base-v2`).

### 4. Vector search

The embeddings are stored in a **FAISS** index. When a user asks a question, the query is converted into an embedding and compared against the stored vectors to find relevant chunks.

### 5. Response generation

The retrieved information is passed to **Llama 2.3 through Ollama**, along with the user's query and available conversation context.

### 6. Agent assistance

The retrieved information and generated response are presented through the Streamlit application as an agent-assist workflow.

## Intent Classification

The project also includes a separate intent classification component using **Logistic Regression**.

The text is transformed using the saved `vectorizer.pkl`, and the trained model in `intent_model.pkl` is used to identify the likely intent of a customer query.

The detected intent can then be tracked as part of the wider conversation workflow.

## Running the Project

Clone the repository:

```bash
git clone <repository-url>
cd RAG_MVP-Agent-Assist
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Or create the Conda environment:

```bash
conda env create -f environment.yml
conda activate <environment-name>
```

Make sure Ollama is installed and the required Llama model is available locally.

Then start the application:

```bash
streamlit run app.py
```

## Project Purpose

This project was built as an MVP to get practical experience with building an end-to-end **RAG and AI application**, rather than only working with individual machine learning models.

It covers the main steps involved in taking unstructured documents, turning them into searchable information, retrieving relevant context, and using that context in an LLM-powered application.

## Future Improvements

Some areas I would like to explore further include:

* Adding retrieval and response evaluation metrics
* Improving intent classification
* Adding document/source references to responses
* Testing different embedding models
* Comparing different retrieval strategies
* Improving conversation memory
* Adding more structured logging and error handling
* Deploying the application as a production service
