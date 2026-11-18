import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


@dataclass
class Node:
    id: int  # position in self.nodes; chunk ids cited in the syntheses are level-0 node ids
    text: str
    level: int  # 0 = original chunk, 1+ = summary
    theme: str = ""  # theme given by the LLM to a summary; empty for original chunks
    children: list[int] = field(default_factory=list)

class RaptorThemeSynthesizer:
    """Theme-level synthesis of a corpus: group the chunks into themes with a RAPTOR tree
    (Sarthi et al., 2024), then write one synthesis per theme by map-reduce over the original chunks.

    Tree: embed texts, reduce them with UMAP, cluster them with a Gaussian mixture (global then local,
    soft membership), summarize each cluster with an LLM, and repeat on the summaries.
    Source: https://github.com/langchain-ai/langchain/blob/v0.1.17rc1/cookbook/RAPTOR.ipynb
    
    Themes: the top-level nodes are merged when near-duplicate; each theme is synthesized from the
    original chunks under it.

    LLM calls go through vLLM: all the clusters of a tree level are sent at once, and each theme
    is synthesized in one call (plus one merge call if needed). The output is constrained by a
    JSON schema (structured outputs), so the format is always respected.
    """

    SUMMARY_PROMPT = (
        "Voici des extraits de documents.\n\n"
        "<extraits>\n{texts}\n</extraits>\n\n"
        "Résume ces extraits en français, uniquement à partir de leur contenu. "
        "Le résumé doit faire environ un quart de la longueur des extraits, entrer "
        "directement dans le sujet sans introduction et conserver tous les faits "
        "utiles : chiffres, dates, noms, titres de documents. N'invente rien.\n\n"
        "Réponds avec deux champs :\n"
        "- `theme` : le thème principal des extraits ;\n"
        "- `summary` : le résumé."
    )
    
    THEME_SYNTHESIS_PROMPT = (
        "Voici des extraits de documents sur le thème : {label}\n"
        "Chaque extrait est précédé de son identifiant entre crochets.\n\n"
        "<extraits>\n{chunks}\n</extraits>\n\n"
        "Rédige en français une synthèse de ce que ces extraits disent sur ce thème, "
        "uniquement à partir de leur contenu. Regroupe ce qui se répète, relie ce qui "
        "se complète et signale ce qui varie ou se contredit d'un document à l'autre. "
        "Conserve tous les faits distincts : chiffres, dates, noms, titres de documents. "
        "N'invente rien.\n\n"
        "Réponds avec trois champs :\n"
        "- `opening` : un paragraphe d'ouverture de 2 à 3 phrases donnant le message d'ensemble ;\n"
        "- `points` : la liste des faits distincts, chacun avec `fact` (le fait) et "
        "`sources` (les identifiants des extraits qui l'appuient) ;\n"
        "- `divergences` : les contradictions entre extraits, c'est-à-dire les cas où au moins deux "
        "extraits donnent des informations incompatibles sur un même point (valeurs, dates, noms "
        "ou affirmations différentes). Qu'un extrait ne mentionne pas un fait n'est PAS une "
        "divergence. Liste vide s'il n'y en a pas, ce qui est le cas le plus fréquent."
    )

    MERGE_PROMPT = (
        "Voici plusieurs synthèses partielles sur le thème : {label}\n"
        "Elles portent sur des extraits différents d'un même ensemble de documents.\n\n"
        "<syntheses>\n{parts}\n</syntheses>\n\n"
        "Fusionne-les en français en une seule synthèse, uniquement à partir de leur contenu. "
        "Rédige un nouveau paragraphe d'ouverture, sans recopier ceux des synthèses partielles. "
        "Regroupe les faits identiques en un seul, en gardant tous leurs identifiants, et conserve "
        "tous les faits distincts : chiffres, dates, noms, titres de documents. Reprends les "
        "divergences des synthèses partielles en regroupant les identiques, et ajoute les "
        "contradictions entre faits de parties différentes. N'invente rien.\n\n"
        "Réponds avec trois champs :\n"
        "- `opening` : un paragraphe d'ouverture de 2 à 3 phrases donnant le message d'ensemble ;\n"
        "- `points` : la liste des faits distincts, chacun avec `fact` (le fait) et "
        "`sources` (les identifiants des extraits qui l'appuient) ;\n"
        "- `divergences` : les contradictions entre extraits, c'est-à-dire les cas où au moins deux "
        "extraits donnent des informations incompatibles sur un même point (valeurs, dates, noms "
        "ou affirmations différentes). Qu'un extrait ne mentionne pas un fait n'est PAS une "
        "divergence. Liste vide s'il n'y en a pas, ce qui est le cas le plus fréquent."
    )

    SUMMARY_SCHEMA = {
        "type": "object",
        "properties": {"theme": {"type": "string"}, "summary": {"type": "string"}},
        "required": ["theme", "summary"],
        "additionalProperties": False,
    }

    SOURCED_FACT_SCHEMA = {
        "type": "object",
        "properties": {
            "fact": {"type": "string"},
            "sources": {"type": "array", "minItems": 1, "items": {"type": "integer"}},
        },
        "required": ["fact", "sources"],
        "additionalProperties": False,
    }

    SYNTHESIS_SCHEMA = {
        "type": "object",
        "properties": {
            "opening": {"type": "string"},
            "points": {"type": "array", "minItems": 1, "items": SOURCED_FACT_SCHEMA},
            "divergences": {"type": "array", "items": SOURCED_FACT_SCHEMA},
        },
        "required": ["opening", "points", "divergences"],
        "additionalProperties": False,
    }

    def __init__(self, embed_model: str | Path, llm_model: str | Path,
                 reduced_dim: int = 10, threshold: float = 0.1, max_clusters: int = 50, local_neighbors: int = 10,
                 n_levels: int = 2, max_input_tokens: int = 16384, max_new_tokens: int = 16384,
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.95,
                 random_state: int = 224) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = LLM(
            model=str(llm_model),
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_input_tokens + max_new_tokens,
            max_num_batched_tokens=max_input_tokens + max_new_tokens,
            max_num_seqs=256,
            trust_remote_code=False,
        )  # vLLM reserves gpu_memory_utilization of the GPU, the embedder uses the rest
        self.tokenizer = self.llm.get_tokenizer()
        self.embedder = SentenceTransformer(str(embed_model), device=self.device)
        self.reduced_dim = reduced_dim
        self.threshold = threshold
        self.max_clusters = max_clusters
        self.local_neighbors = local_neighbors
        self.n_levels = n_levels
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.random_state = random_state

        # output storage
        self.nodes: list[Node] = []
        self.embeddings = np.empty((0, 0))

    ## LLM (vLLM, guided decoding) -----------------------------------------------------------------------------------

    def _generate_json(self, prompts: list[str], schemas: list[dict]) -> list[dict]:
        """Run all prompts in one vLLM batch; each output is constrained by its own JSON schema."""
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        params = [SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens,
                                 structured_outputs=StructuredOutputsParams(json=schema))
                  for schema in schemas]
        outputs = self.llm.chat(messages, params)

        input_lengths = [len(o.prompt_token_ids) for o in outputs]
        print(f"{len(outputs)} requests, input tokens: max {max(input_lengths)}, total {sum(input_lengths)}")

        results = []
        for output in outputs:
            generation = output.outputs[0]
            if generation.finish_reason != "stop":  # truncated JSON cannot be parsed
                tail = generation.text[-3000:]
                raise RuntimeError(f"Generation ended with '{generation.finish_reason}' after "
                                   f"{len(generation.token_ids)} tokens "
                                   f"(input {len(output.prompt_token_ids)} tokens). Tail:\n{tail!r}")
            results.append(json.loads(generation.text))
        return results

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    ## Clustering -----------------------------------------------------------------------------------------------------

    def _gmm_bic(self, reduced: np.ndarray, k: int) -> float:
        """BIC of a k-component Gaussian mixture fitted on the reduced embeddings (lower is better)."""
        return GaussianMixture(k, random_state=self.random_state).fit(reduced).bic(reduced)

    def _gmm_soft_clusters(self, reduced: np.ndarray) -> list[list[int]]:
        """Pick the cluster count by BIC, then keep every text whose probability exceeds the threshold.

        Example with 4 texts, best_k = 3 and threshold = 0.1:
            probs (texts × clusters)        output
            text 0: [0.95, 0.04, 0.01]      [[0, 2],   # cluster 0: texts 0 and 2
            text 1: [0.02, 0.97, 0.01]       [1, 2],   # cluster 1: texts 1 and 2
            text 2: [0.60, 0.35, 0.05]       [3]]      # cluster 2: text 3
            text 3: [0.01, 0.02, 0.97]
        """
        bics = [self._gmm_bic(reduced, k) for k in range(1, min(self.max_clusters, len(reduced)))]
        best_k = int(np.argmin(bics)) + 1  # bics[0] is k=1
        probs = GaussianMixture(best_k, random_state=self.random_state).fit(reduced).predict_proba(reduced)
        return [np.flatnonzero(probs[:, c] > self.threshold).tolist() for c in range(best_k)]

    def _reduce_dimensions(self, embeddings: np.ndarray, n_neighbors: int) -> np.ndarray:
        n_neighbors = max(2, min(n_neighbors, len(embeddings) - 1))  # UMAP needs 2 <= n_neighbors < n
        return umap.UMAP(n_neighbors=n_neighbors, n_components=self.reduced_dim, metric="cosine",
                         random_state=self.random_state).fit_transform(embeddings)

    def _cluster_global_local(self, embeddings: np.ndarray) -> list[list[int]]:
        """Global clustering, then local clustering inside each global cluster. Returns lists of indices."""
        if len(embeddings) <= self.reduced_dim + 1:
            return [list(range(len(embeddings)))]  # too few texts for UMAP dim reduction

        clusters = []
        global_reduced = self._reduce_dimensions(embeddings, int((len(embeddings) - 1) ** 0.5))
        for members in self._gmm_soft_clusters(global_reduced):
            if len(members) == 0:
                continue
            if len(members) <= self.reduced_dim + 1:
                clusters.append(members)  # keep the small global cluster as it is
                continue
            local_reduced = self._reduce_dimensions(embeddings[members], self.local_neighbors)
            for local_positions in self._gmm_soft_clusters(local_reduced):
                if len(local_positions) == 0:
                    continue  # empty local cluster
                # map local positions (within this global cluster) to global positions in `embeddings`
                clusters.append([members[p] for p in local_positions])
        return clusters

    ## Raptor tree summary --------------------------------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts, normalize_embeddings=True, batch_size=8)
    
    def build_tree(self, texts: list[str], cluster_texts: list[str] | None = None) -> None:
        """texts: read by the LLM and indexed; cluster_texts: used to cluster the leaves (default: texts)."""
        self.nodes = [Node(id=i, text=text, level=0) for i, text in enumerate(texts)]
    
        current_nodes = list(self.nodes)  # nodes of the level being clustered
        for level in range(1, self.n_levels + 1):
            # Cluster the current level
            texts_to_embed = (cluster_texts if (cluster_texts and level == 1)
                              else [node.text for node in current_nodes])
            clusters_indices = self._cluster_global_local(self.embed_texts(texts_to_embed))
            print(f"level {level}: {len(current_nodes)} nodes -> {len(clusters_indices)} clusters "
                  f"(sizes: {sorted(len(indices) for indices in clusters_indices)})")
    
            # Summarize all the clusters of the level in one vLLM batch
            clusters_children = [[current_nodes[i] for i in indices] for indices in clusters_indices]
            prompts = []
            for children in clusters_children:
                prompts.append(
                    self.SUMMARY_PROMPT.format(
                        texts="\n\n---\n\n".join([child.text for child in children])
                    )
                )
            summaries = self._generate_json(prompts, [self.SUMMARY_SCHEMA] * len(prompts))

            parent_nodes = []
            for children, summary in zip(clusters_children, summaries):
                parent = Node(id=len(self.nodes), text=summary["summary"].strip(), level=level,
                              theme=summary["theme"].strip(), children=[child.id for child in children])
                self.nodes.append(parent)
                parent_nodes.append(parent)
    
            current_nodes = parent_nodes
            if len(clusters_indices) == 1:
                break  # a single cluster: its summary is the root
    
        self.embeddings = self.embed_texts([node.text for node in self.nodes])

    def search_tree(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Collapsed-tree retrieval: returns (node id, cosine score) over chunks and summaries of every level."""
        scores = self.embeddings @ self.embed_texts([query])[0]
        node_ids_by_score = np.argsort(scores)[::-1]          # node ids sorted by score, best first
        top_ids = node_ids_by_score[:k]                       # keep the k best nodes
        return [(int(i), float(scores[i])) for i in top_ids]  # (node id, score) pairs

    def leaf_chunk_ids(self, node_id: int) -> list[int]:
        """All original chunks under a node (a leaf returns itself)."""
        node = self.nodes[node_id]  # a node's id is its position in self.nodes
        if node.level == 0:
            return [node.id]

        leaves = set()
        for child in node.children:                  # for each child of this summary
            for leaf in self.leaf_chunk_ids(child):  # take all the leaves under that child
                leaves.add(leaf)                     # add them; a set ignores duplicates
        return sorted(leaves)                        # back to a list, in corpus order

    def save_tree(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        nodes_as_dicts = [asdict(n) for n in self.nodes]
        json_text = json.dumps(nodes_as_dicts, ensure_ascii=False, indent=2)
        (out_dir / "nodes.json").write_text(json_text, encoding="utf-8")
        np.save(out_dir / "embeddings.npy", self.embeddings)

    def load_tree(self, in_dir: str | Path) -> None:
        in_dir = Path(in_dir)

        json_text = (in_dir / "nodes.json").read_text(encoding="utf-8")
        self.nodes = [Node(**d) for d in json.loads(json_text)]
        self.embeddings = np.load(in_dir / "embeddings.npy")

        if any(n.id != i for i, n in enumerate(self.nodes)):
            raise ValueError("node ids do not match their position in nodes.json")

        if len(self.embeddings) != len(self.nodes):
            raise ValueError(f"{len(self.nodes)} nodes but {len(self.embeddings)} embeddings: "
                             "the files do not come from the same build")

    ## Themes and synthesis -------------------------------------------------------------------------------------------

    def _group_into_themes(self, level: int, distance_threshold: float) -> list[dict]:
        """Merge near-duplicate nodes of `level` into themes."""
        layer = [n for n in self.nodes if n.level == level]
        if len(layer) < 2:  # a single top-level node is a single theme (AgglomerativeClustering needs 2 samples)
            labels = np.zeros(len(layer), dtype=int)
        else:
            E = self.embeddings[[n.id for n in layer]]
            labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                             distance_threshold=distance_threshold).fit_predict(E)
        out = []
        for lab in sorted(set(labels)):
            members = [layer[j].id for j in np.flatnonzero(labels == lab)]
            leaves = set().union(*(self.leaf_chunk_ids(m) for m in members))
            out.append({"members": members, "leaves": leaves})
        return out

    def _pack(self, texts: list[str]) -> list[list[str]]:
        """Split texts into batches of at most max_input_tokens (an oversized text gets its own batch)."""
        batches, current, size = [], [], 0
        for text in texts:
            n = self._count_tokens(text)
            if current and size + n > self.max_input_tokens:
                batches.append(current)
                current, size = [], 0
            current.append(text)
            size += n
        batches.append(current)
        return batches

    def synthesize_themes(self, out_dir: str | Path | None = None, level: int = 2,
                          distance_threshold: float = 0.25) -> list[dict]:
        """Merge the top-level nodes into themes, then write one synthesis per theme from its original chunks.

        For each theme: its chunks are split into batches synthesized in one vLLM call (map),
        then the partial syntheses are merged if there are several (reduce).
        If out_dir is given, each theme is saved there as theme_<n>.json."""
        themes = self._group_into_themes(level, distance_threshold)
        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        for n, th in enumerate(themes, 1):
            th["label"] = " / ".join(self.nodes[m].theme for m in th["members"])
            leaves = set(th["leaves"])

            # Map: one partial synthesis per batch of chunks
            chunks = [f"[{i}] {self.nodes[i].text}" for i in sorted(leaves)]
            prompts = [self.THEME_SYNTHESIS_PROMPT.format(label=th["label"], chunks="\n\n".join(batch))
                       for batch in self._pack(chunks)]
            partials = self._generate_json(prompts, [self.SYNTHESIS_SCHEMA] * len(prompts))

            # Reduce: merge the partial syntheses if there are several
            if len(partials) > 1:
                parts = "\n\n".join(f"[Partie {k + 1}]\n{json.dumps(p, ensure_ascii=False)}"
                                    for k, p in enumerate(partials))
                merge_prompt = self.MERGE_PROMPT.format(label=th["label"], parts=parts)
                synthesis = self._generate_json([merge_prompt], [self.SYNTHESIS_SCHEMA])[0]
            else:
                synthesis = partials[0]

            # Result
            for item in synthesis["points"] + synthesis["divergences"]:  # drop ids outside the theme
                item["sources"] = [s for s in item["sources"] if s in leaves]
            cited = {s for item in synthesis["points"] + synthesis["divergences"] for s in item["sources"]}
            th["opening"] = synthesis["opening"]
            th["points"] = synthesis["points"]
            th["divergences"] = synthesis["divergences"]
            th["leaves"] = cited
            th["unused"] = sorted(leaves - cited)

            if out_dir:
                theme_json = {"label": th["label"], "members": th["members"], "leaves": sorted(th["leaves"]),
                              "unused": th["unused"], "opening": th["opening"],
                              "points": th["points"], "divergences": th["divergences"]}
                (out_dir / f"theme_{n}.json").write_text(
                    json.dumps(theme_json, ensure_ascii=False, indent=2), encoding="utf-8")
        return themes

    @staticmethod
    def load_themes(in_dir: str | Path) -> list[dict]:
        """Load the themes saved by synthesize_themes, in theme order."""
        files = sorted(Path(in_dir).glob("theme_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        themes = []
        for f in files:
            th = json.loads(f.read_text(encoding="utf-8"))
            th["leaves"] = set(th["leaves"])  # same type as in memory (JSON stores a list)
            themes.append(th)
        return themes