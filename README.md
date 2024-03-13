# RAGnarok

RAGnarok is a [Retrieval-Augmented Generation](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) chatbot frontend for [Nemesis](https://github.com/SpecterOps/Nemesis). It allows you to ask questions about text extracted from compatible documents processed by Nemesis.

## RAG

**Short explanation:** The general idea with Retrieval-Augmented Generation (RAG) is to allow a large language model (LLM) to answer questions about documents you've indexed.

**Medium explanation:** RAG involves processing and turning text inputs into set-length vectors via an embedding model, which are then stored in a backend vector database. Questions to the LLM are then used to look up the "most similiar" chunks of text which are then fed into the context prompt for a LLM.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*kSkeaXRvRzbJ9SrFZaMoOg.png)
[*Source*](https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2)

***Longer explanation in the rest of the section :)***

***Even Longer explanation in [this blog post](https://posts.specterops.io/summoning-ragnarok-with-your-nemesis-7c4f0577c93b).***

#### Indexing

Retrieval-augumented generation is an architecture where documents being processed undergo the following process:

1. Plaintext is extracted from any incoming documents.
   - Nemesis uses [Apache Tika](https://tika.apache.org/) to extract text from compatible documents.
2. The text is tokenized into chunks of up to X tokens, where X depends on the *context window* of the embedding model used.
   - Nemesis uses Langchain's [TokenTextSplitter](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.TokenTextSplitter.html), a chunk size of 510 tokens, and a 15% overlap between chunks.
3. Each chunk of text is processed by an [embedding model](https://huggingface.co/spaces/mteb/leaderboard) which turns the input text into a fixed-length vector of floats.
   - As [Pinecone explains](https://www.pinecone.io/learn/vector-embeddings/), what's cool about embedding models is that the vector representations they produce preserve "semantic similiarity", meaning that more similiar chunks of text will have more similiar vectors.
   - Nemesis currently uses the [TaylorAI/gte-tiny](https://huggingface.co/TaylorAI/gte-tiny) embedding model as it's fast, but others are possible.
4. Each vector and associated snippet of text is stored in a vector database.
   - Nemesis uses Elasticsearch for vector storage. 

#### Semantic Search

This is the initial indexing process that Nemesis has been performing for a while. However, in order to complete a RAG-pipeline, the next steps are:

5. Take an input prompt, such as "*What is a certificate?*" and run it through the same embedding model files were indexed with.
6. Query the vector database (e.g., Elasticsearch) for the nearest **k** vectors + associated text chunks that are "closest" to the prompt input vector.
   - This will return the **k** chunks of text that are the most similiar to the input query.
7. We also use Elasticsearch's traditional(-ish) BM25 text search over the text for each chunk.
   - These two lists of results are combined with [Reciprocal Rank Fusion](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking), and the top results from the fused list are returned.
   - **Note:** steps 6 and 7 happen in the `nlp` container in Nemesis. This is exposed at http://\<nemesis\>/nlp/hybrid_search 

#### Reranking

We now have the **k** most chunks of text most simliar to our input query. If we want to get a bit facier, we can execute what's called [reranking](https://www.pinecone.io/learn/series/rag/rerankers/).

7. With reranking, the the prompt question and text results are paired up (question, text) and fed into a more powerful model (well, more powerful than the embedding model) tuned and known as a reranker. The reranker generates a simliarity score of the input prompt and text chunk.
   - RAGnarok uses an adapted version of [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) for reranking.
8. The results are then **reranked** and the top X number of results are selected.

#### LLM Processing

9. Finally, the resulting texts are combined with a prompt to the (local) LLM. Think something along the lines of "Given these chunks of text {X}, answer this question {Y}".
