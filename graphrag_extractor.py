import json
from dataclasses import asdict, dataclass
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


@dataclass
class EntityNode:
    name: str
    type: str
    description: str
    chunk_id: int  # chunk the entity was extracted from (index in the chunk list)

@dataclass
class Relation:
    source: str  # source entity name
    target: str  # target entity name
    label: str   # short relation name, e.g. "critique"
    description: str
    chunk_id: int


class GraphRAGExtractor:
    """Extract entities and relations from text chunks with a local LLM (first step of GraphRAG).

    Equivalent of the LlamaIndex GraphRAG v1 cookbook's GraphRAGExtractor, without LlamaIndex:
    Source: https://developers.llamaindex.ai/python/examples/cookbooks/graphrag_v1/
    For each chunk, the LLM returns entities (name, type, description) and relations (source, target,
    relation, description) as JSON. Each extracted item keeps the id of its chunk, to cite sources later.
    LLM calls go through vLLM: chunks are sent in batches, and the output is constrained by a JSON schema
    (structured outputs), so the format is always respected.
    """

    EXTRACT_PROMPT = (
        "À partir du texte ci-dessous, extrais au plus {max_knowledge_triplets} relations entre entités.\n"
        "1. Identifie les entités : organismes, personnes, lieux, documents, projets ou dispositifs, notions clés. "
        "Pour chacune : son nom, son type et une description en une phrase.\n"
        "2. Identifie les relations entre ces entités. Pour chacune : l'entité source, l'entité cible, "
        "un nom de relation court (par exemple « finance », « critique », « propose ») et une description "
        "en une phrase.\n"
        "Reprends exactement les noms des entités dans les relations. N'invente rien.\n"
        "Réponds uniquement avec un JSON valide, sans texte autour, au format :\n"
        '{{"entities": [{{"entity_name": "...", "entity_type": "...", "entity_description": "..."}}], '
        '"relationships": [{{"source_entity": "...", "target_entity": "...", "relation": "...", '
        '"relationship_description": "..."}}]}}\n\n'
        "Texte :\n{text}"
    )

    ENTITY_SCHEMA = {
        "type": "object",
        "properties": {
            "entity_name": {"type": "string"},
            "entity_type": {"type": "string"},
            "entity_description": {"type": "string"},
        },
        "required": ["entity_name", "entity_type", "entity_description"],
        "additionalProperties": False,
    }

    RELATION_SCHEMA = {
        "type": "object",
        "properties": {
            "source_entity": {"type": "string"},
            "target_entity": {"type": "string"},
            "relation": {"type": "string"},
            "relationship_description": {"type": "string"},
        },
        "required": ["source_entity", "target_entity", "relation", "relationship_description"],
        "additionalProperties": False,
    }

    def __init__(self, llm_model: str | Path, extract_prompt: str | None = None,
                 max_paths_per_chunk: int = 20, max_input_tokens: int = 8192, max_new_tokens: int = 8192,
                 batch_size: int = 64, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.90) -> None:
        self.llm = LLM(
            model=str(llm_model),
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_input_tokens + max_new_tokens,
            max_num_seqs=256,
            trust_remote_code=False,
        )
        self.extract_prompt = extract_prompt or self.EXTRACT_PROMPT
        self.max_paths_per_chunk = max_paths_per_chunk
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size  # chunks per vLLM call; results are saved after each batch

        # the relation limit of the prompt is also enforced by the schema
        self.extract_schema = {
            "type": "object",
            "properties": {
                "entities": {"type": "array", "maxItems": 2 * max_paths_per_chunk,  # anti-loop guard
                             "items": self.ENTITY_SCHEMA},
                "relationships": {"type": "array", "maxItems": max_paths_per_chunk,
                                  "items": self.RELATION_SCHEMA},
            },
            "required": ["entities", "relationships"],
            "additionalProperties": False,
        }

    def extract_from_llm(self, texts: list[str]) -> list[dict]:
        """Run the extraction prompt on all texts in one vLLM batch; returns the parsed JSON answers."""
        messages = [[{"role": "user",
                      "content": self.extract_prompt.format(text=text, max_knowledge_triplets=self.max_paths_per_chunk)}]
                    for text in texts]
        # for extraction we want faithful, reproducible output, not variety
        params = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens,
                                structured_outputs=StructuredOutputsParams(json=self.extract_schema))
        outputs = self.llm.chat(messages, params)

        answers = []
        for output in outputs:
            generation = output.outputs[0]
            if generation.finish_reason != "stop":  # truncated JSON cannot be parsed
                tail = generation.text[-3000:]
                raise RuntimeError(f"Generation ended with '{generation.finish_reason}' after "
                                   f"{len(generation.token_ids)} tokens "
                                   f"(input {len(output.prompt_token_ids)} tokens). Tail:\n{tail!r}")
            answers.append(json.loads(generation.text))
        return answers

    def extract_batch(self, chunk_ids: list[int], texts: list[str]) -> list[tuple[list[EntityNode], list[Relation]]]:
        """Extract the entities and relations of several chunks in one vLLM call."""
        results = []
        for chunk_id, answer in zip(chunk_ids, self.extract_from_llm(texts)):
            # the schema guarantees the keys; drop items whose names are empty strings
            entities = [EntityNode(e["entity_name"], e["entity_type"], e["entity_description"], chunk_id)
                        for e in answer["entities"] if e["entity_name"]]
            relations = [Relation(r["source_entity"], r["target_entity"], r["relation"],
                                  r["relationship_description"], chunk_id)
                         for r in answer["relationships"] if r["source_entity"] and r["target_entity"]]
            results.append((entities, relations))
        return results

    def extract(self, chunk_id: int, text: str) -> tuple[list[EntityNode], list[Relation]]:
        """Extract the entities and relations of one chunk."""
        return self.extract_batch([chunk_id], [text])[0]

    def extract_all(self, texts: list[str], out_file: str | Path) -> tuple[list[EntityNode], list[Relation]]:
        """Extract from every chunk, batch_size chunks per vLLM call, saving each result as one JSON line;
        already extracted chunks are skipped, so an interrupted run resumes where it stopped (at worst one batch
        is redone). Chunk ids are positions in `texts` (same order as `infos`)."""
        out_file = Path(out_file)
        done = {json.loads(line)["chunk_id"] for line in out_file.open(encoding="utf-8")} if out_file.exists() else set()
        todo = [chunk_id for chunk_id in range(len(texts)) if chunk_id not in done]

        with out_file.open("a", encoding="utf-8") as f:
            for start in range(0, len(todo), self.batch_size):
                chunk_ids = todo[start:start + self.batch_size]
                results = self.extract_batch(chunk_ids, [texts[i] for i in chunk_ids])
                for chunk_id, (entities, relations) in zip(chunk_ids, results):
                    f.write(json.dumps({"chunk_id": chunk_id,
                                        "entities": [asdict(e) for e in entities],
                                        "relations": [asdict(r) for r in relations]}, ensure_ascii=False) + "\n")
                f.flush()  # written to disk after each batch, not at the end
                n_entities = sum(len(e) for e, _ in results)
                n_relations = sum(len(r) for _, r in results)
                print(f"chunks {len(done) + start + len(chunk_ids)}/{len(texts)}: "
                      f"{n_entities} entities, {n_relations} relations in this batch")
        return self.load(out_file)

    @staticmethod
    def load(out_file: str | Path) -> tuple[list[EntityNode], list[Relation]]:
        """Read back every extraction saved by extract_all."""
        entities, relations = [], []
        for line in Path(out_file).open(encoding="utf-8"):
            row = json.loads(line)
            entities += [EntityNode(**e) for e in row["entities"]]
            relations += [Relation(**r) for r in row["relations"]]
        return entities, relations