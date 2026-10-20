import contextlib
from pathlib import Path
from collections.abc import Iterable
from docling_core.types.doc import ImageRefMode
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import ConvertPipelineOptions, PictureDescriptionVlmOptions
from docling.document_converter import DocumentConverter, WordFormatOption


class Docx2DoclingDocument:
    """Convert Word (.docx) files to DoclingDocuments, with classified and described pictures.

    DOCX files go through Docling's automatic pipeline. Unlike a PDF, a DOCX already stores
    its structure (headings, paragraphs, lists, tables), so Docling's Word backend reads it
    directly through the simple pipeline: no layout model, no table structure model, no OCR.
    Rebuilding this ourselves would add complexity without improving the result.

    Pictures: the simple pipeline inherits the enrichment step of Docling's ConvertPipeline,
    so picture classification and picture description (VLM, same prompt style as for PDFs)
    run on the images embedded in the document.
    """

    DEFAULT_IMAGE_PROMPT = (
        "Tu analyses une image extraite d'un document professionnel en français. "
        "1) Indique d'abord le type d'image : schéma ou organigramme, graphique, tableau, logo, "
        "icône, capture d'écran, palette de couleurs, photographie ou autre. "
        "2) Décris ensuite précisément son contenu selon son type (éléments, liens, valeurs, "
        "légendes, couleurs). "
        "3) Recopie enfin tout le texte visible, tel quel, ou écris « Aucun texte visible ». "
        "N'invente rien."
    )

    def __init__(self,
                 artifacts_path: str | Path,
                 vlm_model_path: str | Path,
                 image_prompt: str | None = None,
                 num_threads: int = 4) -> None:
        # Docling loads the description model from "artifacts_path / repo_id":
        # link the local VLM there so the other Docling models stay in artifacts_path
        vlm_model_path = Path(vlm_model_path)
        link = Path(artifacts_path) / vlm_model_path.name
        if not link.exists():
            link.symlink_to(vlm_model_path, target_is_directory=True)

        opts = ConvertPipelineOptions(
            artifacts_path=artifacts_path,
            accelerator_options=AcceleratorOptions(num_threads=num_threads, device=AcceleratorDevice.AUTO),
            do_picture_classification=True,
            do_picture_description=True,
            picture_description_options=PictureDescriptionVlmOptions(
                repo_id=vlm_model_path.name,  # same name as the symlink
                prompt=image_prompt or self.DEFAULT_IMAGE_PROMPT,
            ),
        )
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.DOCX],
            format_options={InputFormat.DOCX: WordFormatOption(pipeline_options=opts)},
        )

    def convert_all(self, paths: Iterable[str | Path], out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for result in self.converter.convert_all(paths, raises_on_error=False):
            name = Path(result.input.file).stem.strip()

            if result.status == ConversionStatus.FAILURE:
                print("FAILED:", name, result.errors)
                continue

            doc = result.document
            with contextlib.chdir(out_dir):  # keeps the image folder next to the files, not nested
                doc.save_as_json(Path(f"{name}.json"), image_mode=ImageRefMode.REFERENCED)
                doc.save_as_markdown(Path(f"{name}.md"), image_mode=ImageRefMode.REFERENCED)
            print("SAVED:", name, "|", len(doc.tables), "tables |", len(doc.pictures), "pictures")