# BioGraphAI

BioGraphAI is an AI-powered document processing and question-answering tool. It allows users to upload PDFs and receive AI-generated responses based on their content. The system extracts, stores, and retrieves document information using advanced AI techniques. It ensures highly relevant, context-aware answers by leveraging ChromaDB, Ollama embeddings, and a cross-encoder for ranking.

## Features

- Upload PDF documents for intelligent analysis
- Efficiently extract and store document content using ChromaDB
- Use Ollama embeddings for optimized text retrieval
- Rank relevant document sections using a cross-encoder
- Generate AI-powered answers from the document’s main character's perspective
- Interactive Streamlit-powered web interface

## Setup and Installation

### Prerequisites

Before setting up BioGraphAI, ensure you have the following installed:

- **Python 3.8+**
- **pip**
- **Ollama** (for embeddings generation)
- **ChromaDB** (for vector storage)
- **Streamlit** (for the web-based interface)

### Clone the Repository

Download and navigate into the project directory:

```bash
git clone https://github.com/yourusername/BioGraphAI.git
cd BioGraphAI
```

### Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Start Ollama and ChromaDB Services

1. **Run Ollama** (Ensure it is available on your system):
   ```bash
   ollama serve
   ```

2. **Start ChromaDB** (To store document embeddings):
   ```bash
   chromadb start --path ./demo-rag-chroma
   ```

## Running the Application

Once the setup is complete, launch BioGraphAI using Streamlit:

```bash
streamlit run app.py
```

### How to Use BioGraphAI

1. Open the Streamlit app in your browser.
2. Upload a PDF document to analyze.
3. Click **Process Document** to extract and store content.
4. Enter your query in the text box.
5. Receive AI-generated answers based on the document’s content.

## Folder Structure

```
BioGraphAI/
│── app.py                  # Main Streamlit application
│── requirements.txt        # Required dependencies
│── README.md               # Project documentation
│── demo-rag-chroma/        # Persistent vector database storage
```

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for any improvements or feature additions.
