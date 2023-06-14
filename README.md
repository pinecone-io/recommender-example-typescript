# Article Recommender

This tutorial demonstrates how to use Pinecone's similarity search to create a simple personalized article or content recommender.

The goal is to create a recommendation engine that retrieves the best article recommendations for each user. When making recommendations with content-based filtering, we evaluate the user’s past behavior and the content items themselves. So in this example, users will be recommended articles that are similar to those they've already read.

```bash
npm install
```

## Importing the Libraries

We'll start by importing the necessary libraries. We'll be using the `@pinecone-database/pinecone` library to interact with Pinecone. We'll also be using the `danfojs-node` library to load the data into an easy to manipulate dataframe. We'll use the `Document` type from Langchain to keep the data structure consistent across the indexing process and retrieval agent.

We'll be using the `Embedder` class found in `embeddings.ts` to embed the data We'll also be using the `cli-progress` library to display a progress bar.

To load the dataset used in the example, we'll be using a utility called `squadLoader.js`.

```typescript
import { Vector, utils } from "@pinecone-database/pinecone";
import { getEnv } from "utils/util.ts";
import { getPineconeClient } from "utils/pinecone.ts";
import cliProgress from "cli-progress";
import { Document } from "langchain/document";
import * as dfd from "danfojs-node";
import { embedder } from "embeddings.ts";
import { SquadRecord, loadSquad } from "./utils/squadLoader.js";
```

## Upload articles

Next, we will prepare data for the Pinecone vector index, and insert it in batches.

The [dataset](https://components.one/datasets/all-the-news-2-news-articles-dataset/) used throughout this example contains 2.7 million news articles and essays from 27 American publications.

Let's download the dataset.

```bash
wget https://www.dropbox.com/s/cn2utnr5ipathhh/all-the-news-2-1.zip -q --show-progress
unzip -q all-the-news-2-1.zip
mkdir data
mv all-the-news-2-1.csv data/.
```

## Create Vector embeddings

Since the dataset could be pretty big, we'll use a generator function that will yield chunks of data to be processed.

```typescript
async function* processInChunks<T, M extends keyof T, P extends keyof T>(
  dataFrame: dfd.DataFrame,
  chunkSize: number,
  metadataFields: M[],
  pageContentField: P
): AsyncGenerator<Document[]> {
  for (let i = 0; i < dataFrame.shape[0]; i += chunkSize) {
    const chunk = await getChunk(dataFrame, i, chunkSize);
    const records = dfd.toJSON(chunk) as T[];
    yield records.map((record: T) => {
      const metadata: Partial<Record<M, T[M]>> = {};
      for (const field of metadataFields) {
        metadata[field] = record[field];
      }
      return new Document({
        pageContent: record[pageContentField] as string,
        metadata,
      });
    });
  }
}
```

For each chunk, the function generates an array of `Document` objects. The function is defined with three type parameters: `T`, `M`, and `P`.

Here are the parameters the function accepts:

- `dataFrame`: This is the DataFrame that the function will process.
- `chunkSize`: This is the number of records that will be processed in each chunk.
- `metadataFields`: This is an array of field names (which are keys of `T`) to be included in the metadata of each `Document`.
- `pageContentField`: This is the field name (which is a key of `T`) to be used for the page content of each `Document`.

Here's what it the function does:

1. It loops over the DataFrame in chunks of size `chunkSize`.
2. For each chunk, it converts the chunk to JSON to get an array of records (of type `T`).
3. Then, for each record in the chunk, it:
   - Creates a `metadata` object that includes the specified metadata fields from the record.
   - Creates a new `Document` with the `pageContent` from the specified field in the record, and the `metadata` object.
4. It then yields an array of the created `Document` objects for the chunk.

The `yield` keyword is used here to produce a value from the generator function. This allows the function to produce a sequence of values over time, rather than computing them all at once and returning them in a single array.

Next we'll create a function that will generate the embeddings and upsert them into Pinecone. We'll use the `processInChunks` generator function to process the data in chunks. We'll also use the `chunkedUpsert` method to insert the embeddings into Pinecone in batches.

```typescript
async function embedAndUpsert(dataFrame: dfd.DataFrame, chunkSize: number) {
  const chunkGenerator = processInChunks(dataFrame, chunkSize);
  const index = pineconeClient.Index(indexName);

  for await (const documents of chunkGenerator) {
    await embedder.embedBatch(
      documents,
      chunkSize,
      async (embeddings: Vector[]) => {
        await chunkedUpsert(index, embeddings, "default");
        progressBar.increment(embeddings.length);
      }
    );
  }
}
```

We'll use the `splitFile` utility function to split the CSV file we downloaded into chunks of 100k parts each. For the purposes of this example, we'll only use the first 100k records.

```typescript
const fileParts = await splitFile("./data/all-the-news-2-1.csv", 1000000);
const firstFile = fileParts[0];
```

Next, we'll load the data into a DataFrame using `loadCSVFile` and to simplify things, we'll also drop all rows which include a null value.

```typescript
const data = await loadCSVFile(firstFile);
const clean = data.dropNa() as dfd.DataFrame;
```

Now we'll create the Pinecone index and kick off the embedding and upserting process.

```typescript
await createIndexIfNotExists(pineconeClient, indexName, 384);
progressBar.start(clean.shape[0], 0);
await embedder.init("Xenova/all-MiniLM-L6-v2");
await embedAndUpsert(clean, 1);
progressBar.stop();
```

## Query the Pinecone Index

We will query the index for the specific users. The users are defined as a set of the articles that they previously read. More specifically, we will define 10 articles for each user, and based on the article embeddings, we will define a unique embedding for the user.
