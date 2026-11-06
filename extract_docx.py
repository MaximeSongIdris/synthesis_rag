import contextlib
from collections.abc import Iterable
from pathlib import Path

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
        "Décris cette image extraite d'un document français, pour qu'une personne qui ne la voit pas "
        "comprenne son contenu. Réponds uniquement en français, sans introduction, sans rien inventer.\n\n"
        "Réponds exactement dans ce format Markdown, sans titre (#) :\n\n"
        "**Type :** schéma, organigramme, graphique, tableau, logo, icône, capture d'écran, "
        "palette de couleurs, photographie ou autre\n\n"
        "**Description :** selon le type :\n"
        "- schéma ou organigramme : une liste à puces, un élément ou un lien par ligne (« A → B »), "
        "dans l'ordre des étapes ;\n"
        "- graphique : le type, les axes, les séries, les valeurs lisibles et la tendance ;\n"
        "- tableau : un tableau Markdown reprenant les colonnes et les lignes ;\n"
        "- palette de couleurs : un tableau Markdown | Couleur | Code |, une ligne par couleur ;\n"
        "- logo ou icône : sa forme et ses couleurs, en une ou deux phrases ;\n"
        "- capture d'écran ou maquette : une liste à puces des zones visibles et de leur contenu ;\n"
        "- photographie : les personnes, objets, lieu, action et ambiance, en quelques phrases.\n\n"
    )

    def __init__(self,
                 artifacts_path: str | Path,
                 vlm_model: str,
                 image_prompt: str | None = None,
                 num_threads: int = 4) -> None:
        # Docling loads the description model from "artifacts_path / repo_id":
        opts = ConvertPipelineOptions(
            artifacts_path=artifacts_path,
            accelerator_options=AcceleratorOptions(num_threads=num_threads, device=AcceleratorDevice.AUTO),
            do_picture_classification=True,
            do_picture_description=True,
            picture_description_options=PictureDescriptionVlmOptions(
                repo_id=vlm_model,
                prompt=image_prompt or self.DEFAULT_IMAGE_PROMPT,
                generation_config={"max_new_tokens": 512, "do_sample": False},  # default truncates long descriptions
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