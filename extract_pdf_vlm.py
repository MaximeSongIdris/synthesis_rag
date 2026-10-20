import contextlib
from pathlib import Path
from collections.abc import Iterable
from docling_core.types.doc import ImageRefMode
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework, InlineVlmOptions, ResponseFormat, TransformersModelType,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


class VlmPdf2DoclingDocument:
    """Convert PDFs to DoclingDocuments with Docling's VLM pipeline (one VLM reads each page image).

    A single vision-language model transcribes each rendered page (text, tables, images),
    and Docling turns its output (DocTags or Markdown) into a DoclingDocument.
    """

    # source: https://github.com/docling-project/docling/blob/main/docling/datamodel/vlm_model_specs.py#L274
    DEFAULT_PAGE_PROMPT = "Convert this page to markdown. Do not miss any text and only output the bare markdown!"

    def __init__(self,
                 artifacts_path: str | Path,
                 vlm_model_path: str | Path,
                 vlm_options: InlineVlmOptions | None = None,
                 num_threads: int = 4) -> None:
        # Docling loads the VLM from "artifacts_path / repo_id":
        # link the local VLM there so the other Docling models stay in artifacts_path
        vlm_model_path = Path(vlm_model_path)
        link = Path(artifacts_path) / vlm_model_path.name
        if not link.exists():
            link.symlink_to(vlm_model_path, target_is_directory=True)

        vlm_options = (vlm_options or InlineVlmOptions(
            repo_id="",  # set below
            prompt=self.DEFAULT_PAGE_PROMPT,
            response_format=ResponseFormat.MARKDOWN,
            inference_framework=InferenceFramework.TRANSFORMERS,
            transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
            max_new_tokens=4096,
        )).model_copy(update={"repo_id": vlm_model_path.name})  # same name as the symlink

        opts = VlmPipelineOptions(
            artifacts_path=artifacts_path,
            accelerator_options=AcceleratorOptions(num_threads=num_threads, device=AcceleratorDevice.AUTO),
            vlm_options=vlm_options,
            generate_page_images=True,  # the VLM reads page images
        )
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=opts)},
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
            print("SAVED:", name, "|", doc.num_pages(), "pages |", len(doc.tables), "tables |", len(doc.pictures), "pictures")