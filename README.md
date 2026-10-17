# kg-gen: Knowledge Graph Generation from Any Text

📄 [**Paper**](https://arxiv.org/abs/2502.09956) | 🐍 [**Package**](https://pypi.org/project/kg-gen/) | 🤖 [**MCP**](https://github.com/stair-lab/kg-gen/tree/main/mcp/) | 🔬 [**Experiments**](https://github.com/stair-lab/kg-gen/tree/main/experiments/) | 👩🏻‍💻 [**Dataset**](https://huggingface.co/datasets/belindamo/wiki_qa_kggen) | 🐦 [**X Updates**](https://x.com/belindmo)

Welcome! `kg-gen` helps you extract knowledge graphs from any plain text using AI. It can process both small and large text inputs, and it can also handle messages in a conversation format.

Why generate knowledge graphs? `kg-gen` is great if you want to:
- Create a graph to assist with RAG (Retrieval-Augmented Generation)
- Create graph synthetic data for model training and testing
- Structure any text into a graph 
- Analyze the relationships between concepts in your source text

We support API-based and local model providers via [LiteLLM](https://docs.litellm.ai/docs/providers), including OpenAI, Ollama, Anthropic, Gemini, Deepseek, and others. We also use [DSPy](https://dspy.ai/) for structured output generation.

- Try it out by running the scripts in [`tests/`](https://github.com/stair-lab/kg-gen/tree/main/tests).
- Instructions to run our KG benchmark MINE are in [`MINE/`](https://github.com/stair-lab/kg-gen/tree/main/experiments/MINE).
- Read the paper: [KGGen: Extracting Knowledge Graphs from Plain Text with Language Models](https://arxiv.org/abs/2502.09956)

## Powered by a model of your choice

Pass in a `model` string to use a model of your choice. Model calls are routed via LiteLLM, and usually LiteLLM goes by the format of `{model_provider}/{model_name}`. See specifically how to format it at [https://docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers).

Examples of models you can pass in:
- `openai/gpt-5`
- `gemini/gemini-2.5-flash`
- `ollama_chat/deepseek-r1:14b`

You may specify a custom API base url with `base_url` ([example here](https://github.com/stair-lab/kg-gen/tree/main/tests/test_custom_api_base.py)).

## Quick start

Install the module:
```bash
pip install kg-gen
```

Then import and use `kg-gen`. You can provide your text input in one of two formats:
1. A single string  
2. A list of Message objects (each with a role and content)

Below are some example snippets:
```python
from kg_gen import KGGen

# Initialize KGGen with optional configuration
kg = KGGen(
  model="openai/gpt-4o",  # Default model
  temperature=0.0,        # Default temperature
  api_key="YOUR_API_KEY"  # Optional if set in environment or using a local model
)

# EXAMPLE 1: Single string with context
text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."
graph_1 = kg.generate(
  input_data=text_input,
  context="Family relationships"
)
# Output: 
# entities={'Linda', 'Ben', 'Andrew', 'Josh'} 
# edges={'is brother of', 'is father of', 'is mother of'} 
# relations={('Ben', 'is brother of', 'Josh'), 
#           ('Andrew', 'is father of', 'Josh'), 
#           ('Linda', 'is mother of', 'Josh')}
```

### Visualizing KGs
```python
KGGen.visualize(graph, output_path, open_in_browser=True)
```

![viz-tool](images/viz-tool.png)

### More Examples - chunking, clustering, passing in a messages array 

```python
# EXAMPLE 2: Large text with chunking and clustering
with open('large_text.txt', 'r') as f:
  large_text = f.read()
  
# Example input text:
# """
# Neural networks are a type of machine learning model. Deep learning is a subset of machine learning
# that uses multiple layers of neural networks. Supervised learning requires training data to learn
# patterns. Machine learning is a type of AI technology that enables computers to learn from data.
# AI, also known as artificial intelligence, is related to the broader field of artificial intelligence.
# Neural nets (NN) are commonly used in ML applications. Machine learning (ML) has revolutionized
# many fields of study.
# ...
# """

graph_2 = kg.generate(
  input_data=large_text,
  chunk_size=5000,  # Process text in chunks of 5000 chars
  cluster=True      # Cluster similar entities and relations
)
# Output:
# entities={'neural networks', 'deep learning', 'machine learning', 'AI', 'artificial intelligence', 
#          'supervised learning', 'unsupervised learning', 'training data', ...} 
# edges={'is type of', 'requires', 'is subset of', 'uses', 'is related to', ...} 
# relations={('neural networks', 'is type of', 'machine learning'),
#           ('deep learning', 'is subset of', 'machine learning'),
#           ('supervised learning', 'requires', 'training data'),
#           ('machine learning', 'is type of', 'AI'),
#           ('AI', 'is related to', 'artificial intelligence'), ...}
# entity_clusters={
#   'artificial intelligence': {'AI', 'artificial intelligence'},
#   'machine learning': {'machine learning', 'ML'},
#   'neural networks': {'neural networks', 'neural nets', 'NN'}
#   ...
# }
# edge_clusters={
#   'is type of': {'is type of', 'is a type of', 'is a kind of'},
#   'is related to': {'is related to', 'is connected to', 'is associated with'
#  ...}
# }

# EXAMPLE 3: Messages array
messages = [
  {"role": "user", "content": "What is the capital of France?"}, 
  {"role": "assistant", "content": "The capital of France is Paris."}
]
graph_3 = kg.generate(input_data=messages)
# Output: 
# entities={'Paris', 'France'} 
# edges={'has capital'} 
# relations={('France', 'has capital', 'Paris')}

# EXAMPLE 4: Combining multiple graphs
text1 = "Linda is Joe's mother. Ben is Joe's brother."

# Input text 2: also goes by Joe."
text2 = "Andrew is Joseph's father. Judy is Andrew's sister. Joseph also goes by Joe."

graph4_a = kg.generate(input_data=text1)
graph4_b = kg.generate(input_data=text2)

# Combine the graphs
combined_graph = kg.aggregate([graph4_a, graph4_b])

# Optionally cluster the combined graph
clustered_graph = kg.cluster(
  combined_graph,
  context="Family relationships"
)
# Output:
# entities={'Linda', 'Ben', 'Andrew', 'Joe', 'Joseph', 'Judy'} 
# edges={'is mother of', 'is father of', 'is brother of', 'is sister of'} 
# relations={('Linda', 'is mother of', 'Joe'),
#           ('Ben', 'is brother of', 'Joe'),
#           ('Andrew', 'is father of', 'Joe'),
#           ('Judy', 'is sister of', 'Andrew')}
# entity_clusters={
#   'Joe': {'Joe', 'Joseph'},
#   ...
# }
# edge_clusters={ ... }
```

## Install from this repository:

Clone this repository and install dependencies using `pip install -e '.[dev]'`. 

You may verify that it works by running `python tests/test_basic.py` from the root directory. This will also generate a nice visualization in `tests/test_basic.html`.

### MCP Server for AI Agents

For AI agents that need persistent memory capabilities:

```bash
# Install and start MCP server
pip install kg-gen
kggen mcp

# Use with Claude Desktop, custom MCP clients, or other AI applications
```

See the [MCP Server documentation](mcp/) for detailed setup and integration instructions.


## Features

### Chunking Large Texts
For large texts, you can specify a `chunk_size` parameter to process the text in smaller chunks:
```python
graph = kg.generate(
  input_data=large_text,
  chunk_size=5000  # Process in chunks of 5000 characters
)
```

### Clustering Similar Entities and Relations
You can cluster similar entities and relations either during generation or afterwards:
```python
# During generation
graph = kg.generate(
  input_data=text,
  cluster=True,
  context="Optional context to guide clustering"
)

# Or after generation
clustered_graph = kg.cluster(
  graph,
  context="Optional context to guide clustering"
)
```

### Multilingual Support (e.g. Spanish)
`kg-gen` can be tuned for non-English text via two parameters, either at `KGGen(...)` init time or per-call in `generate()`/`deduplicate()`:
- `language`: language code (e.g. `"es"`). Selects the language of every prompt sent to the LLM, plus the sentence splitter and lemmatization rules. Defaults to `"en"`.
- `semhash_model_name`: HuggingFace [Model2Vec](https://github.com/MinishLab/model2vec) encoder used by SemHash. For multilingual/Spanish text, use `"minishlab/potion-multilingual-128M"` instead of the English-only SemHash default.

```python
kg = KGGen(
  model="gemini/gemini-2.5-flash",
  retrieval_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # multilingual embeddings for LM-based dedup
  language="es",
  semhash_model_name="minishlab/potion-multilingual-128M",
)

graph = kg.generate("Gabriel García Márquez nació en Aracataca, Colombia. Es el autor de Cien años de soledad.")
```

Currently supported: `"en"` and `"es"`.

#### What `language` affects
| Stage | Effect |
| --- | --- |
| Entity extraction | Instructions and field descriptions of the extraction signature |
| Relation extraction | Instructions, field descriptions, and the few-shot `subject`/`predicate`/`object` examples — these examples are what steer the language the model emits predicates in |
| Chunking | Picks the matching NLTK punkt sentence tokenizer |
| Dedup normalization | spaCy lemmatization: nouns for entities, nouns and verbs for predicates, so conjugation and number variants collapse before matching |
| LLM-based dedup | Instructions of the signature that picks each cluster's alias — i.e. the entity's final name in the graph |

`language` does **not** choose the embedding models. `semhash_model_name` and `retrieval_model` are set separately; leaving them at their English-only defaults while setting `language="es"` will hurt dedup quality.

Prompt files [`prompts/entidades.txt`](src/kg_gen/prompts/entidades.txt) and [`prompts/relaciones.txt`](src/kg_gen/prompts/relaciones.txt) are the Spanish counterparts of `entities.txt`/`relations.txt`, and are used only on the `no_dspy=True` (litellm) path.

#### What `language` deliberately does not affect
The candidate retrieval inside LLM-based dedup (`get_relevant_items`) is left
exactly as upstream kg-gen has it, for every language. Two known weaknesses
were identified and **not** changed:

- BM25 tokenizes with `text.lower().split()`, so it folds neither accents nor
  punctuation: `México` and `Mexico` share no token.
- BM25 and cosine scores are summed at raw scale (`0.5 * bm25 + 0.5 * cosine`).
  BM25Okapi is unbounded while cosine is in [-1, 1], so the "equal weighting"
  is dominated by BM25 — which is also the half that fails on morphological
  variation, exactly what dedup exists to catch.

Both are real, and neither is fixed here on purpose: changing candidate
retrieval would alter which items the LLM ever gets to compare, in every
language. That makes it an algorithmic change to kg-gen rather than a
linguistic adaptation, and it would confound any measurement that attributes a
difference in graph quality to the Spanish adaptation. Note that
`TextNormalizer.fold()`, on the SemHash path that runs first under
`DeduplicateMethod.FULL`, collapses only case variants (`México` / `MÉXICO`).
It keeps diacritics on purpose, since in Spanish they separate different
words (`año` / `ano`, `papá` / `papa`), so an accent omission such as
`Mexico` / `México` is left to SemHash and to this LLM stage. With the
default `potion-base-8M` encoder, whose tokenizer lowercases and strips
accents, SemHash still merges every pair that differs only in case or
accents, distinct words included, and labels the merge with the first form in
sorted order, usually the unaccented one; `potion-multilingual-128M` keeps
them apart.

The SemHash step itself, by contrast, was changed deliberately in the shared
core, for every language. `language="en"` therefore no longer reproduces
upstream kg-gen's SemHash dedup exactly:

- Items whose normalized, singularized forms differ only in case are grouped
  by `fold()` before SemHash (`US` / `us`, `Paris` / `PARIS`), so they merge
  whatever the encoder or threshold. Upstream left them to SemHash and its
  threshold.
- The display form among those variants, and among originals with the same
  normalized form, comes from an explicit rule (`_representative_sort_key`:
  mixed case, then more diacritics, then shortest), not from whichever
  variant happened to arrive last.
- SemHash receives its records `sorted` and runs with `ann_backend="basic"`
  (exact search), instead of in `set` order on the default approximate index,
  so the form that survives a SemHash merge is the first in sorted order.

Upstream's choices depended on the iteration order of a `set`, which changes
with `PYTHONHASHSEED` between processes, so two runs on the same graph could
export different entity names. Applied to every language alike, these changes
make dedup reproducible without favoring any configuration in a
cross-language comparison.

Bugs are treated differently from algorithm changes: the alias bookkeeping
fixes in `deduplicate_cluster` and `_merge_clusters` apply to every language,
so no configuration benefits from them over another.

#### Requirements and fallback behavior
Non-English `language` values require a spaCy model for lemmatization (Spanish uses `es_core_news_md`). Install it separately, since spaCy models aren't installed as regular pip dependencies:
```bash
python -m spacy download es_core_news_md
```

An unsupported `language` code raises `ValueError`, and a supported one whose spaCy model is not installed raises `OSError` with the install command above. Both are checked when `KGGen` is constructed, or when the code is passed as a per-call override — before any LLM calls are made, rather than at deduplication after all extraction calls have been paid for. The check only looks the model up; it is still loaded at deduplication. English uses `inflect` instead of spaCy, so it needs no spaCy model.

#### Adding a language
1. Add the spaCy model to `_SPACY_MODELS` in [`utils/text_normalization.py`](src/kg_gen/utils/text_normalization.py) and install it.
2. Add the punkt language name to `_PUNKT_LANGUAGES` in [`utils/chunk_text.py`](src/kg_gen/utils/chunk_text.py).
3. Add a key to each prompt table in [`steps/_1_get_entities.py`](src/kg_gen/steps/_1_get_entities.py), [`steps/_2_get_relations.py`](src/kg_gen/steps/_2_get_relations.py), and [`utils/llm_deduplicate.py`](src/kg_gen/utils/llm_deduplicate.py). Any table missing the key falls back to English via `pick()` in [`utils/i18n.py`](src/kg_gen/utils/i18n.py), so a partial translation degrades rather than breaking.
4. Optionally add `prompts/<entities>.txt` and `prompts/<relations>.txt` for the `no_dspy=True` path.
5. Add the code to `SUPPORTED_LANGUAGES` in [`utils/i18n.py`](src/kg_gen/utils/i18n.py). Do this last: it is what makes the code accepted at all, so registering it before the steps above turns a clear "unsupported language" error into silent English fallbacks.

### Aggregating Multiple Graphs
You can combine multiple graphs using the aggregate method:
```python
graph1 = kg.generate(input_data=text1)
graph2 = kg.generate(input_data=text2)
combined_graph = kg.aggregate([graph1, graph2])
```

### Message Array Processing
When processing message arrays, kg-gen:
1. Preserves the role information from each message
2. Maintains message order and boundaries
3. Can extract entities and relationships:
   - Between concepts mentioned in messages
   - Between speakers (roles) and concepts
   - Across multiple messages in a conversation

For example, given this conversation:
```python
messages = [
  {"role": "user", "content": "What is the capital of France?"},
  {"role": "assistant", "content": "The capital of France is Paris."}
]
```

The generated graph might include entities like:
- "user"
- "assistant" 
- "France"
- "Paris"

And relations like:
- (user, "asks about", "France")
- (assistant, "states", "Paris")
- (Paris, "is capital of", "France")

### Citing
If you find KGGen useful, please consider citing:

```
@misc{mo2025kggenextractingknowledgegraphs,
      title={KGGen: Extracting Knowledge Graphs from Plain Text with Language Models}, 
      author={Belinda Mo and Kyssen Yu and Joshua Kazdan and Joan Cabezas and Proud Mpala and Lisa Yu and Chris Cundy and Charilaos Kanatsoulis and Sanmi Koyejo},
      year={2025},
      eprint={2502.09956},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.09956}, 
}
```
## License
The MIT License.
