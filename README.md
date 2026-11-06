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

# download models
docling-tools models download
docling-tools models download easyocr --easyocr-lang fr
cd $HOME/.cache/docling/models
git clone https://huggingface.co/microsoft/harrier-oss-v1-0.6b
git clone https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct

pip install --user graspologic
```
