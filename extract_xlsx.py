import torch
import pandas as pd
from pathlib import Path
from collections.abc import Iterable
from transformers import AutoProcessor, AutoModelForImageTextToText
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel


class Xlsx2DoclingDocument:
    """Convert Excel files to DoclingDocuments, with a description of each table.

    The table itself is extracted by Docling's DocumentConverter (XLSX backend), not rebuilt
    by hand. This is deliberate: Docling reads the workbook structure directly (rows, columns,
    merged cells, hence multi-level column categories) and produces the table cells.
    Rebuilding it ourselves from pandas would mean handling multi-level headers, column spans
    and value types by hand, with a higher risk of errors.

    pandas is only used to read the first raw rows of each sheet (header=None, no header
    interpretation). These rows are given to a vision-language model (text only), which
    writes a description in French, attached to the table as its caption. The full table
    remains the one produced by Docling.
    """

    DEFAULT_PROMPT = (
        "Voici les premières lignes brutes d'une feuille Excel (« {sheet} »). "
        "Les premières lignes sont des en-têtes, parfois sur "
        "plusieurs niveaux : une catégorie de colonnes peut couvrir plusieurs colonnes, "
        "et n'apparaît alors qu'au-dessus de la première.\n\n"
        "{table}\n\n"
        "Décris en français, en 3 à 5 phrases : le sujet du tableau, les catégories de colonnes, "
        "le rôle de chaque colonne et ce que représente une ligne. "
        "N'invente rien, ne donne aucun total ni statistique."
    )

    def __init__(self, vlm_model: str | Path, n_head: int = 10, max_new_tokens: int = 400) -> None:
        self.converter = DocumentConverter(allowed_formats=[InputFormat.XLSX])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(vlm_model)
        self.model = AutoModelForImageTextToText.from_pretrained(vlm_model).to(self.device)
        self.n_head = n_head
        self.max_new_tokens = max_new_tokens

    def describe(self, head: pd.DataFrame, sheet: str) -> str:
        prompt = self.DEFAULT_PROMPT.format(
            sheet=sheet, table=head.fillna("").to_string(header=False, index=False))
        chat = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def convert_all(self, paths: Iterable[str | Path], out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for result in self.converter.convert_all(paths, raises_on_error=False):
            path = Path(result.input.file)
            name = path.stem.strip()

            if result.status == ConversionStatus.FAILURE:
                print("FAILED:", name, result.errors)
                continue

            doc = result.document
            heads = pd.read_excel(path, sheet_name=None, header=None, nrows=self.n_head, dtype=object)
            for sheet_no, (sheet, head) in enumerate(heads.items(), start=1):
                table = next((t for t in doc.tables if t.prov and t.prov[0].page_no == sheet_no), None)
                if table is None:
                    continue
                caption = doc.add_text(label=DocItemLabel.CAPTION, text=self.describe(head, sheet))
                table.captions.append(caption.get_ref())

            doc.save_as_json(out_dir / f"{name}.json")
            doc.save_as_markdown(out_dir / f"{name}.md")
            print("SAVED:", name, "|", len(doc.tables), "tables")