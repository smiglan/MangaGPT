# MangaGPT

MangaGPT lets you chat with your favourite manga chapters. The project
downloads chapter pages, indexes them with OpenAI embeddings and exposes a
simple question–answer interface through Streamlit. Out of the box the
repository knows about several series (One Piece, Naruto, Boruto and JJK), and
you can add more by editing `manga_config.json`.

## Features

* **Automated scraping** – `build_vector_store.py` discovers new chapter pages
  and appends them to a local FAISS index.
* **Chat interface** – `app.py` provides a minimal Streamlit UI for asking
  questions about the indexed content using OpenAI's `gpt-4o-mini` model.
* **Incremental updates** – rerunning the build script only embeds chapters
  that have not been processed before.

## Getting started

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. Provide your OpenAI API key

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Build the vector store for the desired chapters

    ```bash
    # available keys come from manga_config.json
    python build_vector_store.py --manga naruto
    ```

    The script discovers all available chapters for the chosen manga and
    stores them under `vector_store/<manga>/`. Reruns only embed chapters that
    haven't been processed before and include basic retry logic to handle
    flaky network requests.

4. Launch the app

    ```bash
    streamlit run app.py
    ```

    Pick a manga from the dropdown, enter a question such as *"Who joined the
    Straw Hats in chapter 5?"* and the application will respond using the
    indexed chapters.

## Notes

The vector store is stored in the `vector_store/` directory and can grow
quickly. The folder is ignored by Git. Remove it if you want to rebuild the
index from scratch.

All network requests are made against publicly available pages. Respect the
terms of service of the sites you scrape and avoid overwhelming their servers.

## Future work

The current implementation operates on text summaries of chapters. A future
iteration will process raw page images directly, enabling richer questions
about artwork or sound effects. Potential approaches include:

- **Optical character recognition** with tools like Tesseract to extract text
  from scans before embedding.
- **Vision-language models** such as GPT-4o with vision or CLIP to generate
  embeddings from the images themselves.
- **Hybrid pipelines** that combine OCR for dialogue with image encoders for
  panels to provide both textual and visual context.
