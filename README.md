# synthesis_rag

## Activate software environment on Jean Zay

```bash
# source runs the bash command directly on current process
source jz_setup_env.sh
```

## Install missing packages

After activating the environment, install the additional Python packages (they go to the project's `.local` via `PYTHONUSERBASE`):

```bash
pip install --user --no-cache-dir docling[easyocr,vlm]
docling-tools models download
docling-tools models download easyocr --easyocr-lang en --easyocr-lang fr
#pip install --user --no-cache-dir langchain umap-learn scikit-learn langchain_community tiktoken langchain-openai langchainhub chromadb langchain-anthropic
```
