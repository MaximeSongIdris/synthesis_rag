import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import umap
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from transformers import AutoModelForImageTextToText, AutoProcessor


@dataclass
class Node:
    text: str
    level: int  # 0 = original chunk, 1+ = summary
    children: list[int] = field(default_factory=list)

class Raptor:
    """RAPTOR (Sarthi et al., 2024), adapted from the LangChain cookbook to local models.

    Source: https://github.com/langchain-ai/langchain/blob/v0.1.17rc1/cookbook/RAPTOR.ipynb

    Builds a tree bottom-up: embed texts, reduce them with UMAP, cluster them with a Gaussian mixture
    (global then local, soft membership), summarize each cluster with an LLM, and repeat on the summaries.
    Retrieval searches all nodes at once ("collapsed tree").
    """

    SUMMARY_PROMPT = (
        "Voici des extraits de documents. Rédige en français un résumé succinct, de 6 phrases au maximum, "
        "qui donne le sujet commun et les informations essentielles (faits, chiffres, décisions, documents cités). "
        "Sois bref : moins de phrases si le contenu le permet. N'invente rien, sans introduction.\n\n{texts}"
    )

    def __init__(self, embed_model: str | Path, llm_model: str | Path,
                 reduced_dim: int = 10, threshold: float = 0.1, max_clusters: int = 50, local_neighbors: int = 10,
                 n_levels: int = 3, max_new_tokens: int = 220, random_state: int = 224) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # use config
        self.embedder = SentenceTransformer(str(embed_model), device=self.device)
        self.processor = AutoProcessor.from_pretrained(llm_model)
        self.llm = AutoModelForImageTextToText.from_pretrained(llm_model, dtype="auto").to(self.device)
        self.reduced_dim = reduced_dim
        self.threshold = threshold
        self.max_clusters = max_clusters
        self.local_neighbors = local_neighbors
        self.n_levels = n_levels
        self.max_new_tokens = max_new_tokens
        self.random_state = random_state

        # output storage
        self.nodes: list[Node] = []
        self.embeddings = np.empty((0, 0))

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts, normalize_embeddings=True)

    def _bic(self, reduced: np.ndarray, k: int) -> float:
        """BIC of a k-component Gaussian mixture fitted on the reduced embeddings (lower is better)."""
        return GaussianMixture(k, random_state=self.random_state).fit(reduced).bic(reduced)

    def gmm_cluster(self, reduced: np.ndarray) -> list[list[int]]:
        """Pick the cluster count by BIC, then keep every text whose probability exceeds the threshold.
        
        Example with 4 texts, best_k = 3 and threshold = 0.1:
            probs (texts × clusters)        output
            text 0: [0.95, 0.04, 0.01]      [[0, 2],   # cluster 0: texts 0 and 2
            text 1: [0.02, 0.97, 0.01]       [1, 2],   # cluster 1: texts 1 and 2
            text 2: [0.60, 0.35, 0.05]       [3]]      # cluster 2: text 3
            text 3: [0.01, 0.02, 0.97]
        """
        bics = [self._bic(reduced, k) for k in range(1, min(self.max_clusters, len(reduced)))]
        best_k = int(np.argmin(bics)) + 1  # bics[0] is k=1
        probs = GaussianMixture(best_k, random_state=self.random_state).fit(reduced).predict_proba(reduced)
        return [np.flatnonzero(probs[:, c] > self.threshold).tolist() for c in range(best_k)]

    def _reduce(self, embeddings: np.ndarray, n_neighbors: int) -> np.ndarray:
        n_neighbors = max(2, min(n_neighbors, len(embeddings) - 1))  # UMAP needs 2 <= n_neighbors < n
        return umap.UMAP(n_neighbors=n_neighbors, n_components=self.reduced_dim, metric="cosine",
                         random_state=self.random_state).fit_transform(embeddings)

    def cluster(self, embeddings: np.ndarray) -> list[list[int]]:
        """Global clustering, then local clustering inside each global cluster. Returns lists of indices."""
        if len(embeddings) <= self.reduced_dim + 1:
            return [list(range(len(embeddings)))]  # too few texts for UMAP dim reduction

        clusters = []
        for members in self.gmm_cluster(self._reduce(embeddings, int((len(embeddings) - 1) ** 0.5))):
            if len(members) == 0:
                continue
            if len(members) <= self.reduced_dim + 1:
                clusters.append(members)  # keep the small global cluster as it is
                continue
            local_clusters = self.gmm_cluster(self._reduce(embeddings[members], self.local_neighbors))
            for local_positions in local_clusters:
                if len(local_positions) == 0:
                    continue  # empty local cluster
                # map local positions (within this global cluster) to global positions in `embeddings`
                clusters.append([members[p] for p in local_positions])
        return clusters

    def summarize(self, texts: list[str]) -> str:
        prompt = self.SUMMARY_PROMPT.format(texts="\n\n---\n\n".join(texts))
        chat = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        chat_prompt = self.processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = self.processor(text=chat_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # for summaries we want faithful, reproducible text, not variety
            output_ids = self.llm.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]  # drop the prompt tokens
        return self.processor.decode(new_ids, skip_special_tokens=True).strip()

    def build(self, texts: list[str], cluster_texts: list[str] | None = None) -> None:
        """texts: read by the LLM and indexed; cluster_texts: used to cluster the leaves (default: texts)."""
        self.nodes = [Node(text=t, level=0) for t in texts]
    
        curr_layer_node_idx = list(range(len(self.nodes)))
        for level in range(1, self.n_levels + 1):
            # Clusterize the current layer
            to_embed = cluster_texts if (cluster_texts and level == 1) else [self.nodes[i].text for i in curr_layer_node_idx]
            clusters = self.cluster(self.embed(to_embed))
            print(f"level {level}: {len(curr_layer_node_idx)} nodes -> {len(clusters)} clusters "
                  f"(sizes: {sorted(len(c) for c in clusters)})")

            # Build the next layer from clusters
            next_layer_node_idx = []
            for members in clusters:
                children_idx = [curr_layer_node_idx[j] for j in members] 
                summary = self.summarize([self.nodes[i].text for i in children_idx])
                self.nodes.append(Node(text=summary, level=level, children=children_idx))
                next_layer_node_idx.append(len(self.nodes) - 1)
            curr_layer_node_idx = next_layer_node_idx
            if len(clusters) == 1:
                break  # a single cluster: its summary is the root
        self.embeddings = self.embed([n.text for n in self.nodes])

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Collapsed-tree retrieval: returns (node id, cosine score) over chunks and summaries of every level."""
        scores = self.embeddings @ self.embed([query])[0]
        node_ids_by_score = np.argsort(scores)[::-1]          # node ids sorted by score, best first
        top_ids = node_ids_by_score[:k]                       # keep the k best nodes
        return [(int(i), float(scores[i])) for i in top_ids]  # (node id, score) pairs

    def leaf_ids(self, node_id: int) -> list[int]:
        """All original chunks under a node (a leaf returns itself)."""
        node = self.nodes[node_id]
        if node.level == 0:
            return [node_id]

        leaves = set()
        for child in node.children:              # for each child of this summary
            for leaf in self.leaf_ids(child):    # take all the leaves under that child (recursive call)
                leaves.add(leaf)                 # add them; a set ignores duplicates
        return sorted(leaves)                    # back to a list, in corpus order

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # save tree as a json
        nodes_as_dicts = [asdict(n) for n in self.nodes]                       # 1. Node -> dict
        json_text = json.dumps(nodes_as_dicts, ensure_ascii=False, indent=2)   # 2. list of dicts -> JSON text
        path = out_dir / "nodes.json"                                          # 3. file path
        path.write_text(json_text, encoding="utf-8")                           # 4. write the text to the file

        # save embedding
        np.save(out_dir / "embeddings.npy", self.embeddings)

    def load(self, in_dir: str | Path) -> None:
        in_dir = Path(in_dir)

        # read the tree back from the json (reverse of save)
        json_text = (in_dir / "nodes.json").read_text(encoding="utf-8")
        nodes_as_dicts = json.loads(json_text)
        self.nodes = [Node(**d) for d in nodes_as_dicts]  # dict -> Node

        # read the embeddings back
        self.embeddings = np.load(in_dir / "embeddings.npy")

        if len(self.embeddings) != len(self.nodes):
            raise ValueError(f"{len(self.nodes)} nodes but {len(self.embeddings)} embeddings: "
                             "the files do not come from the same build")