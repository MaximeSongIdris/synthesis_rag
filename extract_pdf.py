import contextlib
from collections.abc import Iterable
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode


class Pdf2DoclingDocument:
    """Convert native PDFs to DoclingDocuments (headings, tables, captioned pictures).

    Unlike jpg files, where we build the DoclingDocument ourselves, PDFs go through Docling's
    automatic pipeline. A PDF page mixes text, headings, tables and images, and rebuilding the
    layout analysis, reading order, table structure and picture extraction by hand would be
    far more complex than configuring the existing pipeline.

    Known limitations observed on our corpus:
      - Picture classification and description are not reliable: some images are not
        classified or described, and descriptions can be inaccurate.
      - Table extraction has issues: some tables come out incomplete or badly structured,
        so important tables should be checked against the original PDF.
    """

    DEFAULT_IMAGE_PROMPT = (
        "Décris cette image extraite d'un document français, pour qu'une personne qui ne la voit pas "
        "comprenne son contenu. Réponds uniquement en français, sans introduction, sans rien inventer.\n\n"
        "Commence par le type d'image : schéma ou organigramme, graphique, tableau, logo, icône, "
        "capture d'écran, palette de couleurs, photographie, ou autre. Puis décris selon le type :\n"
        "- schéma ou organigramme : chaque élément, chaque lien (« A vers B ») et l'ordre des étapes ;\n"
        "- graphique : le type, les axes, les séries, les valeurs lisibles et la tendance ;\n"
        "- tableau : les colonnes et le contenu des lignes ;\n"
        "- logo ou icône : sa forme, ses couleurs et le texte qu'il contient, en une ou deux phrases ;\n"
        "- palette de couleurs : le nom et le code de chaque couleur ;\n"
        "- capture d'écran ou maquette : les zones visibles et leur contenu ;\n"
        "- photographie : les personnes, objets, lieu, action et ambiance.\n"
        "Termine par le texte visible, recopié exactement dans l'ordre de lecture, "
        "ou « Aucun texte visible »."
    )

    def __init__(
        self,
        artifacts_path: str | Path,
        vlm_model_path: str | Path,
        image_prompt: str | None = None,
        ocr_langs: list[str] | None = None,
        num_threads: int = 4,
    ) -> None:
        # Docling loads the description model from "artifacts_path / repo_id":
        # link the local VLM there so the other Docling models stay in artifacts_path
        vlm_model_path = Path(vlm_model_path)
        link = Path(artifacts_path) / vlm_model_path.name
        if not link.exists():
            link.symlink_to(vlm_model_path, target_is_directory=True)

        opts = PdfPipelineOptions(artifacts_path=artifacts_path)

        # recover PDF heading levels
        # source: https://docling-project.github.io/docling/usage/advanced_options/#recover-pdf-heading-levels
        opts.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
        opts.generate_parsed_pages = True  # required by the font-style signal

        # rebuild tables as real rows and columns instead of loose text
        # source: https://docling-project.github.io/docling/reference/pipeline_options/#docling.datamodel.pipeline_options.TableStructureOptions.do_cell_matching
        opts.do_table_structure = True
        opts.table_structure_options = TableStructureOptions(do_cell_matching=True)
        opts.table_structure_options.mode = TableFormerMode.ACCURATE  # more accurate TableFormer model

        # extract images from the pdf, describe them for the retrieval system and the LLM
        # source: https://docling-project.github.io/docling/_generated/examples/export_figures/
        # source: https://docling-project.github.io/docling/usage/enrichments/#picture-classification
        opts.generate_picture_images = True
        opts.images_scale = 2
        opts.do_picture_classification = True
        opts.do_picture_description = True
        opts.picture_description_options = PictureDescriptionVlmOptions(
            repo_id="Qwen3-VL-8B-Instruct",
            prompt=image_prompt or self.DEFAULT_IMAGE_PROMPT,
        )

        # use the GPU if available, otherwise CPU threads
        # source: https://docling-project.github.io/docling/_generated/examples/custom_convert/
        opts.accelerator_options = AcceleratorOptions(num_threads=num_threads, device=AcceleratorDevice.AUTO)

        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
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