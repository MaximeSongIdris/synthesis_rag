from __future__ import annotations  # EntityNode and Relation type hints are not evaluated at import

import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from graspologic.partition import hierarchical_leiden
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


class GraphRAGStore:
    """Build the knowledge graph from extracted mentions, detect communities and synthesize them.

    Equivalent of the LlamaIndex GraphRAG v1 cookbook's GraphRAGStore, without LlamaIndex:
    Source: https://developers.llamaindex.ai/python/examples/cookbooks/graphrag_v1/
    Mentions of the same entity (same name, case-insensitive) become one node; repeated relations between
    the same two entities become one weighted edge. Communities: hierarchical Leiden (graspologic), as in the
    cookbook and the GraphRAG paper; the finest level is kept. One LLM synthesis per community (title, opening,
    points and divergences citing chunk ids), written from the community's entities and relations, all
    communities in one vLLM batch. The output is constrained by a JSON schema (structured outputs).
    """

    SYNTHESIS_PROMPT = (
        "Voici des entités et des relations extraites d'un corpus de documents. Elles forment un même thème.\n"
        "Chaque ligne est précédée, entre crochets, des identifiants des extraits d'où elle provient.\n\n"
        "<entites>\n{entities}\n</entites>\n\n"
        "<relations>\n{relations}\n</relations>\n\n"
        "Rédige en français une synthèse de ce thème, uniquement à partir de ces lignes : les acteurs "
        "principaux, ce qu'ils font et les points saillants (décisions, critiques, chiffres). Regroupe ce qui "
        "se répète et relie ce qui se complète. N'invente rien.\n\n"
        "Réponds avec quatre champs :\n"
        "- `title` : un titre court pour le thème ;\n"
        "- `opening` : un paragraphe d'ouverture de 2 à 3 phrases donnant le message d'ensemble ;\n"
        "- `points` : les faits importants, chacun avec `fact` (le fait, en une phrase) et `sources` "
        "(les identifiants des extraits des lignes qui l'appuient) ;\n"
        "- `divergences` : les contradictions entre lignes. Une divergence existe seulement si deux "
        "affirmations ne peuvent pas être vraies en même temps : des valeurs, dates ou noms différents pour "
        "la même chose, ou des positions opposées sur une même question. Des valeurs identiques ne sont pas "
        "une divergence. Ne répète jamais une divergence. Ne sont PAS des divergences : qu'une ligne ne "
        "mentionne pas un fait ; qu'un acteur approuve, reprend, confirme ou complète ce que dit un autre ; "
        "une simple différence de formulation. Liste vide s'il n'y en a pas."
    )

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
            "title": {"type": "string"},
            "opening": {"type": "string"},
            "points": {"type": "array", "minItems": 1, "items": SOURCED_FACT_SCHEMA},
            "divergences": {"type": "array", "maxItems": 30, "items": SOURCED_FACT_SCHEMA},  # max: anti-loop guard
        },
        "required": ["title", "opening", "points", "divergences"],
        "additionalProperties": False,
    }

    def __init__(self, entities: list[EntityNode], relations: list[Relation], llm_model: str | Path,
                 max_cluster_size: int = 20, max_input_tokens: int = 8192, max_new_tokens: int = 4096,
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.90,
                 random_state: int = 224) -> None:
        # LLM for community syntheses
        self.llm = LLM(
            model=str(llm_model),
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_input_tokens + max_new_tokens,
            max_num_seqs=256,
            trust_remote_code=False,
        )
        self.max_new_tokens = max_new_tokens

        # clustering config
        self.max_cluster_size = max_cluster_size  # Leiden splits any community bigger than this
        self.random_state = random_state

        # output storage
        self.graph = nx.Graph()
        self.communities: list[set[str]] = []  # entity keys of each community, biggest first
        self.syntheses: list[dict] = []        # one LLM synthesis per community, same order
        self._build_graph(entities, relations)

    def _generate_json(self, prompts: list[str], schema: dict) -> list[dict]:
        """Run all prompts in one vLLM batch; each output is constrained by the JSON schema."""
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        # for syntheses we want faithful, reproducible text, not variety
        params = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens,
                                structured_outputs=StructuredOutputsParams(json=schema))
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

    @staticmethod
    def key(name: str) -> str:
        """Node id of an entity, insensitive to case."""
        return name.strip().lower()

    def _build_graph(self, entities: list[EntityNode], relations: list[Relation]) -> None:
        """One node per entity, one weighted edge per pair of related entities."""
        # group entities that have the same keys
        mentions: dict[str, list[EntityNode]] = defaultdict(list)
        for e in entities:
            mentions[self.key(e.name)].append(e)

        # combine entities with same keys into a single entity in the knowledge graph
        for entity_key, entity_mentions in mentions.items():
            types = Counter(mention.type for mention in entity_mentions)
            self.graph.add_node(
                entity_key,
                name=entity_mentions[0].name,      # name as first written
                type=types.most_common(1)[0][0],   # most frequent type
                descriptions=list(dict.fromkeys(m.description for m in entity_mentions if m.description)),  # unique, in order
                chunk_ids=sorted({m.chunk_id for m in entity_mentions}),  # all chunks mentioning the entity
            )

        # add relations as edges in the graph
        for r in relations:
            source, target = self.key(r.source), self.key(r.target)
            if source not in self.graph or target not in self.graph or source == target:
                continue  # relation to an entity that was not extracted, or to itself: skipped
            if not self.graph.has_edge(source, target):
                self.graph.add_edge(source, target, weight=0, labels=[], descriptions=[], chunk_ids=[])
            edge = self.graph.edges[source, target]
            edge["weight"] += 1  # how many times this relation was extracted
            edge["labels"].append(r.label)
            if r.description and r.description not in edge["descriptions"]:
                edge["descriptions"].append(r.description)
            if r.chunk_id not in edge["chunk_ids"]:
                edge["chunk_ids"].append(r.chunk_id)

    def build_communities(self) -> None:
        """Hierarchical Leiden on the weighted graph; each entity keeps its finest community."""
        connected = self.graph.subgraph([n for n in self.graph if self.graph.degree(n) > 0])  # no isolated entity
        clusters = hierarchical_leiden(connected, max_cluster_size=self.max_cluster_size,
                                       random_seed=self.random_state)

        # get only the final communities
        groups: dict[int, set[str]] = defaultdict(set)
        for c in clusters:
            if c.is_final_cluster:
                groups[c.cluster].add(c.node)
        self.communities = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)
        print(f"{len(self.communities)} communities (sizes: {[len(c) for c in self.communities]})")

    @staticmethod
    def _ids(chunk_ids: list[int], max_ids: int = 10) -> str:
        """Chunk ids shown in front of a line (at most max_ids, to keep hub entities short)."""
        return "[" + ", ".join(map(str, sorted(chunk_ids)[:max_ids])) + "]"

    def describe_community(self, community: set[str], max_lines: int = 60) -> tuple[str, str]:
        """Entity and relation lines of a community, each with its chunk ids, most connected first,
        to stay within the prompt budget."""
        nodes = sorted(community, key=self.graph.degree, reverse=True)[:max_lines]
        entity_lines = [f"{self._ids(self.graph.nodes[n]['chunk_ids'])} {self.graph.nodes[n]['name']} "
                        f"({self.graph.nodes[n]['type']}) : {' '.join(self.graph.nodes[n]['descriptions'][:2])}"
                        for n in nodes]

        edges = sorted(self.graph.subgraph(community).edges(data=True),
                       key=lambda edge: edge[2]["weight"], reverse=True)[:max_lines]
        relation_lines = [f"{self._ids(d['chunk_ids'])} {self.graph.nodes[s]['name']} -> {self.graph.nodes[t]['name']} -> "
                          f"{Counter(d['labels']).most_common(1)[0][0]} -> {' '.join(d['descriptions'][:2])}"
                          for s, t, d in edges]
        return "\n".join(entity_lines), "\n".join(relation_lines)

    def synthesize_communities(self) -> None:
        """One LLM synthesis per community, all communities in one vLLM batch."""
        prompts = []
        for community in self.communities:
            entities, relations = self.describe_community(community)
            prompts.append(self.SYNTHESIS_PROMPT.format(entities=entities, relations=relations))
        self.syntheses = self._generate_json(prompts, self.SYNTHESIS_SCHEMA) if prompts else []

        for community, synthesis in zip(self.communities, self.syntheses):
            allowed = set(self.community_chunks(community))
            for key in ("points", "divergences"):  # drop duplicate ids and ids outside the community
                for item in synthesis[key]:
                    item["sources"] = [i for i in dict.fromkeys(item["sources"]) if i in allowed]
                synthesis[key] = [item for item in synthesis[key] if item["sources"]]

    def community_chunks(self, community: set[str]) -> list[int]:
        """Chunks where the community's entities are mentioned, to cite its sources with `infos`."""
        return sorted({i for n in community for i in self.graph.nodes[n]["chunk_ids"]})

    def save(self, out_dir: str | Path) -> None:
        """Save the graph and the communities with their syntheses."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        graph_data = nx.node_link_data(self.graph, edges="edges")
        (out_dir / "graph.json").write_text(json.dumps(graph_data, ensure_ascii=False, indent=2), encoding="utf-8")
        communities = [{"title": syn["title"], "entities": sorted(c), "chunk_ids": self.community_chunks(c),
                        "opening": syn["opening"], "points": syn["points"], "divergences": syn["divergences"]}
                       for c, syn in zip(self.communities, self.syntheses)]
        (out_dir / "communities.json").write_text(json.dumps(communities, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")