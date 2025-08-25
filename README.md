# MangaGPT

This project downloads manga chapter pages, builds a vector store using
OpenAI embeddings and exposes a simple question–answer interface through
Streamlit.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Provide your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Build the vector store for your chapter links:

   ```bash
   python build_vector_store.py
   ```

   Adjust the `links` list in the script as needed.

4. Launch the app:

   ```bash
   streamlit run app.py
   ```

You can then ask questions against the downloaded chapters using the
`gpt-5-nano` model.
