"""Gradio viewer for the theme syntheses: click a point or a divergence to see the source chunks it cites.

Reads the theme_<n>.json files written by RaptorThemeSynthesizer.synthesize_themes (and split_themes):
    {"title", "members", "leaves", "unused", "opening",
     "points": [{"fact", "sources"}], "divergences": [{"fact", "sources"}],
     "sub_themes": [{"title", "opening", "points", "divergences"}], "sub_unused"}
A table of contents (themes and their sub-themes, titles only) sits at the top; clicking a title shows that
theme, section by section.

Usage:
    python synthesis_app.py --themes outputs/themes --nodes outputs/nodes.json
"""

import argparse
import html
import json
import os
import random
from functools import partial
from pathlib import Path

import gradio as gr

PLACEHOLDER = "<p style='opacity:.7'>Click a point or a divergence on the left to see the chunks it cites.</p>"

CSS = """
.gradio-container { width: 100% !important; max-width: 1440px !important; margin-left: auto; margin-right: auto; }
.point-btn, .point-btn button { text-align: left !important; justify-content: flex-start !important;
                                white-space: normal !important; font-weight: 400 !important; }
.divergence-btn, .divergence-btn button { border-left: 3px solid var(--color-accent) !important; }
.opening { font-size: 1.05em; line-height: 1.5; }
.sub-opening { opacity: .8; font-style: italic; }
.tree-btn, .tree-btn button { text-align: left !important; justify-content: flex-start !important;
                              background: none !important; border: none !important; box-shadow: none !important;
                              padding: 2px 6px !important; min-height: 0 !important; white-space: normal !important; }
.tree-btn:hover, .tree-btn button:hover { text-decoration: underline; }
.tree-theme, .tree-theme button { font-weight: 600 !important; margin-top: 6px !important; }
.tree-sub, .tree-sub button { padding-left: 28px !important; font-weight: 400 !important; opacity: .85; }
#contents { gap: 0 !important; }
"""


def load_themes(themes_dir: Path) -> list[dict]:
    files = sorted(themes_dir.glob("theme_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]

def load_chunks(nodes_path: Path) -> dict[int, str]:
    """Original chunks only (level 0), keyed by node id."""
    nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
    return {i: n["text"] for i, n in enumerate(nodes) if n["level"] == 0}

def theme_title(theme: dict) -> str:
    return theme.get("title") or theme.get("label", "")  # "label": files written before titles existed

def select(theme_idx: int, sub_idx: int | None) -> tuple[int, int | None]:
    """Value stored when a title of the table of contents is clicked."""
    return theme_idx, sub_idx

## HTML -----------------------------------------------------------------------------------------------------------

def chunk_card(chunk_id: int, text: str | None) -> str:
    body = html.escape(text) if text is not None else "<i>Chunk not found in nodes.json.</i>"
    return (
        "<div style='border:1px solid var(--border-color-primary);border-radius:8px;"
        "padding:10px 12px;margin-bottom:10px;background:var(--background-fill-secondary)'>"
        f"<div style='font-weight:600;margin-bottom:6px'>[{chunk_id}]</div>"
        f"<div style='white-space:pre-wrap;font-size:.92em'>{body}</div></div>"
    )

def cards(chunks: dict[int, str], ids: list[int]) -> str:
    ids = list(dict.fromkeys(ids))  # unique, original order
    if not ids:
        return "<p style='opacity:.7'>No chunk cited.</p>"
    return "".join(chunk_card(i, chunks.get(i)) for i in ids)

def item_html(chunks: dict[int, str], item: dict, context: str = "", divergence: bool = False) -> str:
    """A point or a divergence: where it comes from, its text, then the chunks it cites."""
    kind = "Divergence" if divergence else "Point"
    n = len(set(item["sources"]))
    header = (f"<p style='opacity:.7;margin-bottom:4px'>{kind}{' · ' + html.escape(context) if context else ''}"
              f" · {n} source{'s' if n > 1 else ''}</p>"
              f"<p><b>{html.escape(item['fact'])}</b></p>")
    return header + cards(chunks, item["sources"])

def chunks_html(chunks: dict[int, str], ids: list[int], header: str) -> str:
    return f"<p><b>{html.escape(header)}</b></p>" + cards(chunks, ids)

## App ------------------------------------------------------------------------------------------------------------

def item_buttons(items: list[dict], chunks: dict[int, str], sources: gr.HTML, context: str,
                 divergence: bool = False) -> None:
    """One clickable button per point (or divergence); a click shows its chunks on the right."""
    for item in items:
        n = len(set(item["sources"]))
        button = gr.Button(f"{'⚠ ' if divergence else ''}{item['fact']}  ({n})", size="sm",
                           variant="secondary",
                           elem_classes=["point-btn"] + (["divergence-btn"] if divergence else []))
        button.click(partial(item_html, chunks, item, context, divergence), None, sources)

def synthesis_block(block: dict, chunks: dict[int, str], sources: gr.HTML, context: str) -> None:
    """Points, then divergences, of a theme or a sub-theme."""
    item_buttons(block.get("points", []), chunks, sources, context)
    if block.get("divergences"):
        gr.Markdown(f"**Divergences ({len(block['divergences'])})**")
        item_buttons(block["divergences"], chunks, sources, context, divergence=True)

def build_app(themes: list[dict], chunks: dict[int, str]) -> gr.Blocks:
    with gr.Blocks(title="Synthesis explorer",
                   theme=gr.themes.Default(primary_hue="red", secondary_hue="pink", neutral_hue="sky")) as demo:
        gr.HTML(f"<style>{CSS}</style>")  # in the page, so it works whatever the Gradio version
        gr.HTML("<center><h1>Synthesis explorer</h1></center>")
        selection = gr.State((0, None))  # (theme index, sub-theme index or None)
        sources = gr.HTML(PLACEHOLDER, render=False)

        # Table of contents: themes and their sub-themes, titles only
        with gr.Accordion("Contents", open=True):
            with gr.Column(elem_id="contents"):
                for t, theme in enumerate(themes):
                    button = gr.Button(f"{t + 1}. {theme_title(theme)}", size="sm",
                                       elem_classes=["tree-btn", "tree-theme"])
                    button.click(partial(select, t, None), None, selection)
                    for k, sub in enumerate(theme.get("sub_themes", [])):
                        button = gr.Button(f"{t + 1}.{k + 1}  {sub['title']}", size="sm",
                                           elem_classes=["tree-btn", "tree-sub"])
                        button.click(partial(select, t, k), None, selection)

        with gr.Row():
            with gr.Column(scale=1):
                @gr.render(inputs=selection)
                def show_theme(selected: tuple[int, int | None]):
                    idx, open_sub = selected
                    theme = themes[idx]
                    title = theme_title(theme)
                    sub_themes = theme.get("sub_themes", [])
                    points = theme.get("points", [])

                    gr.Markdown(f"## {idx + 1}. {title}")
                    if theme.get("opening"):
                        gr.Markdown(theme["opening"], elem_classes="opening")
                    gr.Markdown(f"<span style='opacity:.7'>{len(theme.get('leaves', []))} chunks cited · "
                                f"{len(points)} point{'s' if len(points) > 1 else ''}"
                                + (f" · {len(sub_themes)} sub-themes" if sub_themes else "") + "</span>")

                    if sub_themes:
                        # One section per sub-theme
                        for k, sub in enumerate(sub_themes):
                            n = len(sub.get("points", []))
                            label = f"{k + 1}. {sub['title']}  ({n} point{'s' if n > 1 else ''})"
                            with gr.Accordion(label, open=(k == open_sub if open_sub is not None else k == 0)):
                                if sub.get("opening"):
                                    gr.Markdown(sub["opening"], elem_classes="sub-opening")
                                synthesis_block(sub, chunks, sources, context=sub["title"])
                    else:
                        synthesis_block(theme, chunks, sources, context=title)

                    # Coverage: the chunks left out by the synthesis, then by the sub-themes, shown directly
                    unused, sub_unused = theme.get("unused", []), theme.get("sub_unused", [])
                    if unused or sub_unused:
                        with gr.Accordion(f"Coverage ({len(unused) + len(sub_unused)} chunks left out)", open=False):
                            if unused:
                                gr.HTML(chunks_html(chunks, unused,
                                                    f"{len(unused)} chunk(s) of the theme not cited by the synthesis"))
                            if sub_unused:
                                gr.HTML(chunks_html(chunks, sub_unused,
                                                    f"{len(sub_unused)} chunk(s) cited by the synthesis but by no sub-theme"))

            with gr.Column(scale=1, elem_id="sources-panel"):
                gr.Markdown("### Source chunks")
                sources.render()

        selection.change(lambda: PLACEHOLDER, None, sources)
    return demo

def launch_gradio(themes_dir: Path, nodes_path: Path) -> None:
    """Build the app and launch it on JupyterHub (same way as HalChat)."""
    interface = build_app(load_themes(themes_dir), load_chunks(nodes_path))

    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
    if prefix is None:  # not on JupyterHub (local test)
        interface.launch()
        return

    server = "https://jupyterhub.idris.fr"
    port = random.randint(1111, 2222)  # a fixed port value can create a ports conflict
    root_path = f"{prefix}proxy/{port}/"
    print("Gradio Interface available on :", server + root_path)
    interface.launch(root_path=root_path, server_port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--themes", type=Path, required=True, help="folder with theme_<n>.json")
    parser.add_argument("--nodes", type=Path, required=True, help="nodes.json saved by save_tree")
    args = parser.parse_args()
    launch_gradio(args.themes, args.nodes)