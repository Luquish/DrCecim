# DrCecim

## Description

DrCecim is a virtual assistant specifically designed for students of the Faculty of Medicine at the University of Buenos Aires (UBA). This chatbot provides accurate answers to administrative queries, procedures, information about approved courses, and institutional regulations.

Unlike a medical assistant, DrCecim focuses exclusively on resolving doubts related to administrative and academic processes of the faculty, using official UBA documentation as its source of information.

## Main Features

- **Responses based on official documents**: Uses RAG (Retrieval-Augmented Generation) technology to provide accurate information extracted from official UBA documents.
- **User-friendly interface**: Developed with Streamlit to offer an intuitive user experience.
- **Natural language processing**: Understands natural language queries to facilitate interaction with students.
- **Vectorized knowledge base**: Efficiently stores and retrieves information from institutional documents.

## Technologies Used

- **Streamlit**: For the user interface
- **LangChain**: Framework for AI applications
- **Groq**: Language model for response generation
- **Cohere**: For reranking and improving results
- **ChromaDB**: Vector database for storing processed documents
- **HuggingFace Embeddings**: For text vectorization

## Requirements

### API Keys
- GROQ_API_KEY
- COHERE_API_KEY

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/DrCecim.git
cd DrCecim
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
   - Option 1: Configure keys in Streamlit Secrets
   - Option 2: Create a `config.json` file with your API keys

4. Run the application:
```bash
streamlit run main.py
```

## Project Structure

- `main.py`: Main Streamlit application
- `vectorize_documents.py`: Script for processing and vectorizing documents
- `data/`: Directory for storing official UBA documents
- `vector_db_dir/`: Storage for the vector database
- `.streamlit/`: Streamlit configuration

## Usage

1. Start the application with `streamlit run main.py`
2. Ask questions about administrative procedures, regulations, or academic information from the UBA Faculty of Medicine
3. The system will search through official documentation and provide accurate answers

## License

This project is licensed under [include license].

## Contributions

Contributions are welcome. If you wish to contribute:
1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request