"""Gradio viewer for the theme syntheses: click a point or a divergence to see the source chunks it cites.

Reads the theme_<n>.json files written by RaptorThemeSynthesizer.synthesize_themes:
    {"label", "members", "leaves", "unused", "opening",
     "points": [{"fact", "sources"}],
     "divergences": [{"fact", "sources"}]}

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


def load_themes(themes_dir: Path) -> list[dict]:
    files = sorted(themes_dir.glob("theme_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]

def load_chunks(nodes_path: Path) -> dict[int, str]:
    """Original chunks only (level 0), keyed by node id."""
    nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
    return {i: n["text"] for i, n in enumerate(nodes) if n["level"] == 0}

## HTML -----------------------------------------------------------------------------------------------------------

def truncate(text: str, max_len: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1] + "…"

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

def item_html(chunks: dict[int, str], item: dict, prefix: str = "") -> str:
    """A point or a divergence: its text, then the chunks it cites."""
    return f"<p><b>{prefix}{html.escape(item['fact'])}</b></p>" + cards(chunks, item["sources"])

def unused_html(chunks: dict[int, str], ids: list[int]) -> str:
    header = (f"<p><b>{len(ids)} chunk(s) of this theme not cited by the synthesis</b></p>")
    return header + cards(chunks, ids)

## App ------------------------------------------------------------------------------------------------------------

def build_app(themes: list[dict], chunks: dict[int, str]) -> gr.Blocks:
    with gr.Blocks(title="Synthesis explorer",
                   theme=gr.themes.Default(primary_hue="red", secondary_hue="pink", neutral_hue="sky")) as demo:
        gr.HTML("""
        <style>
        /* Limit width and center the app content */
        .gradio-container {
            max-width: 1440px;
            margin-left: auto;
            margin-right: auto;
        }
        /* Left-align long button labels */
        .point-btn button, button.point-btn { text-align: left; justify-content: flex-start; white-space: normal; }
        </style>
        """)
        gr.HTML("<center><h1>Synthesis explorer</h1></center>")
        theme_choice = gr.Dropdown(
            choices=[(f"{i + 1}. {t['label']}", i) for i, t in enumerate(themes)],
            value=0, label="Theme", interactive=True,
        )
        sources = gr.HTML(PLACEHOLDER, render=False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Synthesis")

                @gr.render(inputs=theme_choice)
                def show_theme(idx: int):
                    theme = themes[idx]
                    points = theme.get("points", [])
                    divergences = theme.get("divergences", [])
                    unused = theme.get("unused", [])

                    if theme.get("opening"):
                        gr.Markdown(theme["opening"])

                    gr.Markdown(f"#### Points ({len(points)})")
                    for point in points:
                        button = gr.Button(truncate(point["fact"]), size="sm", elem_classes="point-btn")
                        button.click(partial(item_html, chunks, point), None, sources)

                    if divergences:
                        gr.Markdown(f"#### Divergences ({len(divergences)})")
                        for divergence in divergences:
                            button = gr.Button(truncate(divergence["fact"]), size="sm",
                                               variant="secondary", elem_classes="point-btn")
                            button.click(partial(item_html, chunks, divergence, "Divergence : "), None, sources)

                    if unused:
                        gr.Markdown("#### Coverage")
                        button = gr.Button(f"{len(unused)} uncited chunk(s) out of "
                                           f"{len(unused) + len(theme.get('leaves', []))}", size="sm",
                                           variant="secondary")
                        button.click(partial(unused_html, chunks, unused), None, sources)

            with gr.Column(scale=1):
                gr.Markdown("### Source chunks")
                sources.render()

        theme_choice.change(lambda: PLACEHOLDER, None, sources)
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