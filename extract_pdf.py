import contextlib
from collections.abc import Iterable
from pathlib import Path

from docling_core.types.doc import ImageRefMode
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    HeadingHierarchyOptions,
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


class Pdf2DoclingDocument:
    """Convert native PDFs to DoclingDocuments (headings, tables, captioned pictures, OCR).

    Unlike jpg files, where we build the Docling
    Document ourselves, PDFs go through Docling's
    automatic pipeline. A PDF page mixes text, headings, tables and images, and rebuilding the
    layout analysis, reading order, table structure and picture extraction by hand would be
    far more complex than configuring the existing pipeline.

    Known limitations observed on our corpus:
      - Picture classification and description are not reliable: some images are not
        classified or described, and descriptions can be inaccurate.
      - Table extraction has issues: some tables come out incomplete or badly structured,
        so important tables should be checked against the original PDF.
      - OCR does not seem to extract the text inside images. In the default OCR mode, OCR
        runs on regions found by the layout model that lack native PDF text, so text in an
        image that is not detected as its own region, or that overlaps native text, may be
        missed.
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

    def __init__(
        self,
        artifacts_path: str | Path,
        vlm_model: str,
        image_prompt: str | None = None,
        ocr_langs: list[str] | None = None,
        num_threads: int = 4,
    ) -> None:
        # Docling loads the description model from "artifacts_path / repo_id":
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
        opts.do_picture_classification = False  # classification with the picture description model
        opts.do_picture_description = True
        opts.picture_description_options = PictureDescriptionVlmOptions(
            repo_id=vlm_model,
            prompt=image_prompt or self.DEFAULT_IMAGE_PROMPT,
            generation_config={"max_new_tokens": 512, "do_sample": False},  # default truncates long descriptions
        )

        # TODO: check if ocr is working
        # read text that exists only as pixels (labels inside diagrams, scanned inserts)
        # source: https://docling-project.github.io/docling/_generated/examples/custom_convert/
        opts.do_ocr = True
        opts.ocr_options = EasyOcrOptions(lang=ocr_langs or ["fr"])

        # use the GPU if available, otherwise CPU threads
        # source: https://docling-project.github.io/docling/_generated/examples/custom_convert/
        opts.accelerator_options = AcceleratorOptions(num_threads=num_threads, device=AcceleratorDevice.AUTO)

        self.converter = DocumentConverter(
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