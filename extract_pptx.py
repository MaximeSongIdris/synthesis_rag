from collections.abc import Iterable
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter


class SimplePptx2DoclingDocument:
    """Convert text-only pptx files to DoclingDocuments with Docling's default PPTX converter."""

    def __init__(self) -> None:
        self.converter = DocumentConverter(allowed_formats=[InputFormat.PPTX])

    def convert_all(self, paths: Iterable[str | Path], out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for result in self.converter.convert_all(paths, raises_on_error=False):
            name = Path(result.input.file).stem.strip()

            if result.status == ConversionStatus.FAILURE:
                print("FAILED:", name, result.errors)
                continue

            doc = result.document
            doc.save_as_json(out_dir / f"{name}.json")
            doc.save_as_markdown(out_dir / f"{name}.md")
            print("SAVED:", name, "|", len(doc.pages), "slides")