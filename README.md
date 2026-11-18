# TextBook Assistant RAG + LangChain

Upload a text-based PDF, inspect its chunks and vectors, search for a study topic, and read an extractive answer with links to the original PDF pages.

Run `python -m pip install -r requirements.txt` and `./run.sh`, then open http://127.0.0.1:8001/.

The index runs locally and combines TF-IDF with latent semantic analysis when the document has sufficient vocabulary and passages. It returns verbatim excerpts from the uploaded book. PDF page numbers can differ from printed page numbers. Scanned pages require OCR before upload. The limits are 500 MB, 5,000 pages, and 100,000 chunks. The server keeps up to eight recent indexes until it restarts.

In Context Generation, **Exact textbook sentences** works without a model. **Generated explanation** uses a local Ollama model. Start Ollama with `nemotron-3-nano:4b` installed and launch the app with `./run.sh`. Click a citation to highlight its exact quote and open the original PDF page.

### Hybrid retrieval

Run `ollama pull all-minilm` once. Select **Hybrid search with local embeddings** when indexing a PDF. The app embeds passages in batches on the local Ollama server, then combines keyword and embedding rankings. Fusion scores express rank agreement and are not probabilities. Leave the option unchecked to index without Ollama. Hybrid indexing takes longer on large PDFs and requires re-uploading books indexed with the earlier search mode.


The hosted site reads PDFs on the student's device using PDF.js. It offers local
keyword search and cited excerpts without uploading the textbook. It downloads
PDF.js from jsDelivr on first use. Browser processing requires sufficient memory.
The local Python application saves textbooks and indexes in .textbooks/ and
supports Nemotron through Ollama. Browser indexes currently last for the open tab.
