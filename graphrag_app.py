"""Gradio viewer for the GraphRAG communities: click a point, a divergence or an entity to see its source chunks.

Reads the files written by GraphRAGStore.save in `out_dir`:
    communities.json: [{"title", "entities", "chunk_ids", "opening",
                        "points": [{"fact", "sources"}], "divergences": [{"fact", "sources"}]}]
    graph.json (optional): entity names, types, descriptions and chunk ids
The chunk texts are the `texts` list the extraction ran on (chunk id = position in the list); `sources`
(ChunkSource / LeafInfo list, same order) adds the document name and pages to each chunk card.

Usage from a notebook:
    from graphrag_app import launch_gradio
    launch_gradio("data/INJA/Livrable 1/graphrag", texts, sources)

Usage from a terminal (chunk texts read from the RAPTOR nodes.json, level 0):
    python graphrag_app.py --dir "data/INJA/Livrable 1/graphrag" --nodes "data/INJA/Livrable 1/raptor/nodes.json"
"""

import argparse
import html
import json
import os
import random
from functools import partial
from pathlib import Path

import gradio as gr

PLACEHOLDER = "<p style='opacity:.7'>Click a point, a divergence or an entity on the left to see its source chunks.</p>"

CSS = """
.gradio-container { width: 100% !important; max-width: 1440px !important; margin-left: auto; margin-right: auto; }
.point-btn, .point-btn button { text-align: left !important; justify-content: flex-start !important;
                                white-space: normal !important; font-weight: 400 !important; }
.divergence-btn, .divergence-btn button { border-left: 3px solid var(--color-accent) !important; }
.entity-btn, .entity-btn button { flex: 0 0 auto !important; min-width: 0 !important; font-weight: 400 !important; }
.opening { font-size: 1.05em; line-height: 1.5; }
.tree-btn, .tree-btn button { text-align: left !important; justify-content: flex-start !important;
                              background: none !important; border: none !important; box-shadow: none !important;
                              padding: 2px 6px !important; min-height: 0 !important; white-space: normal !important; }
.tree-btn:hover, .tree-btn button:hover { text-decoration: underline; }
#contents { gap: 0 !important; }
"""

## Data -----------------------------------------------------------------------------------------------------------

def load_communities(out_dir: Path) -> list[dict]:
    return json.loads((out_dir / "communities.json").read_text(encoding="utf-8"))

def load_entities(out_dir: Path) -> dict[str, dict]:
    """Graph nodes by entity key ({} if graph.json is missing)."""
    path = out_dir / "graph.json"
    if not path.exists():
        return {}
    graph = json.loads(path.read_text(encoding="utf-8"))
    return {node["id"]: node for node in graph["nodes"]}

def load_texts_from_nodes(nodes_path: Path) -> list[str]:
    """Chunk texts from a RAPTOR nodes.json: the level-0 nodes, in order (their ids are the chunk ids)."""
    nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
    return [n["text"] for n in nodes if n["level"] == 0]

def source_label(source) -> str:
    """'document · p. 3, 4' for a ChunkSource / LeafInfo (object or dict); '' if unknown."""
    if source is None:
        return ""
    get = source.get if isinstance(source, dict) else lambda key, default=None: getattr(source, key, default)
    document, pages = get("document", ""), get("pages", []) or []
    return f"{document}{' · p. ' + ', '.join(map(str, pages)) if pages else ''}"

## HTML -----------------------------------------------------------------------------------------------------------

def chunk_card(chunk_id: int, texts: list[str], sources: list | None) -> str:
    text = texts[chunk_id] if 0 <= chunk_id < len(texts) else None
    body = html.escape(str(text)) if text is not None else "<i>Chunk not found.</i>"
    label = source_label(sources[chunk_id]) if sources and 0 <= chunk_id < len(sources) else ""
    return (
        "<div style='border:1px solid var(--border-color-primary);border-radius:8px;"
        "padding:10px 12px;margin-bottom:10px;background:var(--background-fill-secondary)'>"
        f"<div style='font-weight:600;margin-bottom:6px'>[{chunk_id}]"
        f"{' <span style=\"font-weight:400;opacity:.7\">· ' + html.escape(label) + '</span>' if label else ''}</div>"
        f"<div style='white-space:pre-wrap;font-size:.92em'>{body}</div></div>"
    )

def cards(ids: list[int], texts: list[str], sources: list | None) -> str:
    ids = list(dict.fromkeys(ids))  # unique, original order
    if not ids:
        return "<p style='opacity:.7'>No chunk cited.</p>"
    return "".join(chunk_card(i, texts, sources) for i in ids)

def item_html(texts: list[str], sources: list | None, item: dict, context: str, divergence: bool = False) -> str:
    """A point or a divergence: where it comes from, its text, then the chunks it cites."""
    n = len(set(item["sources"]))
    return (f"<p style='opacity:.7;margin-bottom:4px'>{'Divergence' if divergence else 'Point'} · "
            f"{html.escape(context)} · {n} source{'s' if n > 1 else ''}</p>"
            f"<p><b>{html.escape(item['fact'])}</b></p>" + cards(item["sources"], texts, sources))

def entity_html(texts: list[str], sources: list | None, entity: dict) -> str:
    """An entity: its type, its descriptions, then the chunks that mention it."""
    descriptions = "".join(f"<li>{html.escape(d)}</li>" for d in entity.get("descriptions", []))
    ids = entity.get("chunk_ids", [])
    return (f"<p style='opacity:.7;margin-bottom:4px'>Entity · {html.escape(entity.get('type', ''))} · "
            f"{len(ids)} chunk{'s' if len(ids) > 1 else ''}</p>"
            f"<p><b>{html.escape(entity.get('name', entity.get('id', '')))}</b></p>"
            + (f"<ul>{descriptions}</ul>" if descriptions else "") + cards(ids, texts, sources))

## App ------------------------------------------------------------------------------------------------------------

def select(idx: int) -> int:
    """Value stored when a title of the table of contents is clicked."""
    return idx

def build_app(communities: list[dict], entities: dict[str, dict], texts: list[str],
              sources: list | None = None) -> gr.Blocks:
    with gr.Blocks(title="Community explorer",
                   theme=gr.themes.Default(primary_hue="red", secondary_hue="pink", neutral_hue="sky")) as demo:
        gr.HTML(f"<style>{CSS}</style>")  # in the page, so it works whatever the Gradio version
        gr.HTML("<center><h1>Community explorer</h1></center>")
        selection = gr.State(0)  # index of the community shown
        panel = gr.HTML(PLACEHOLDER, render=False)

        # Table of contents: community titles
        with gr.Accordion("Contents", open=True):
            with gr.Column(elem_id="contents"):
                for i, community in enumerate(communities):
                    n = len(community["entities"])
                    button = gr.Button(f"{i + 1}. {community['title']}  ({n} entit{'ies' if n > 1 else 'y'})",
                                       size="sm", elem_classes="tree-btn")
                    button.click(partial(select, i), None, selection)

        with gr.Row():
            with gr.Column(scale=1):
                @gr.render(inputs=selection)
                def show_community(idx: int):
                    community = communities[idx]
                    title = community["title"]
                    points, divergences = community.get("points", []), community.get("divergences", [])

                    gr.Markdown(f"## {idx + 1}. {title}")
                    if community.get("opening"):
                        gr.Markdown(community["opening"], elem_classes="opening")
                    n = len(community["entities"])
                    gr.Markdown(f"<span style='opacity:.7'>{n} entit{'ies' if n > 1 else 'y'} · "
                                f"{len(community.get('chunk_ids', []))} chunks · {len(points)} "
                                f"point{'s' if len(points) > 1 else ''}</span>")

                    # Entities, most mentioned first; a click shows their descriptions and chunks
                    keys = sorted(community["entities"],
                                  key=lambda k: -len(entities.get(k, {}).get("chunk_ids", [])))
                    with gr.Accordion(f"Entities ({len(keys)})", open=False):
                        with gr.Row():
                            for key in keys:
                                entity = entities.get(key, {"id": key, "name": key})
                                button = gr.Button(entity.get("name", key), size="sm", variant="secondary",
                                                   elem_classes="entity-btn")
                                button.click(partial(entity_html, texts, sources, entity), None, panel)

                    gr.Markdown(f"#### Points ({len(points)})")
                    for point in points:
                        button = gr.Button(point["fact"], size="sm", variant="secondary", elem_classes="point-btn")
                        button.click(partial(item_html, texts, sources, point, title), None, panel)

                    if divergences:
                        gr.Markdown(f"#### Divergences ({len(divergences)})")
                        for divergence in divergences:
                            button = gr.Button(f"⚠ {divergence['fact']}", size="sm", variant="secondary",
                                               elem_classes=["point-btn", "divergence-btn"])
                            button.click(partial(item_html, texts, sources, divergence, title, True), None, panel)

            with gr.Column(scale=1):
                gr.Markdown("### Source chunks")
                panel.render()

        selection.change(lambda: PLACEHOLDER, None, panel)
    return demo

def launch_gradio(out_dir: str | Path, texts: list[str], sources: list | None = None) -> None:
    """Build the app and launch it on JupyterHub (same way as HalChat)."""
    out_dir = Path(out_dir)
    interface = build_app(load_communities(out_dir), load_entities(out_dir), texts, sources)

    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
    if prefix is None:  # not on JupyterHub (local test)
        interface.launch(show_error=True)
        return

    server = "https://jupyterhub.idris.fr"
    port = random.randint(1111, 2222)  # a fixed port value can create a ports conflict
    root_path = f"{prefix}proxy/{port}/"
    print("Gradio Interface available on :", server + root_path)
    interface.launch(root_path=root_path, server_port=port, show_error=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True, help="folder with communities.json (and graph.json)")
    parser.add_argument("--nodes", type=Path, required=True, help="RAPTOR nodes.json: its level-0 texts are the chunks")
    args = parser.parse_args()
    launch_gradio(args.dir, load_texts_from_nodes(args.nodes))