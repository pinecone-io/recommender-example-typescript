import { getEnv } from "utils/util.ts";
import { getPineconeClient } from "utils/pinecone.ts";
import { embedder } from "embeddings.ts";
import * as dfd from "danfojs-node";

const indexName = getEnv("PINECONE_INDEX");
const pineconeClient = await getPineconeClient();

const pineconeIndex = pineconeClient.Index(indexName);

await embedder.init("Xenova/all-MiniLM-L6-v2");

// We create a "dummy" query to build a user with an interest in "Sports"
const query = "~~~";
const queryEmbedding = await embedder.embed(query);

const queryResult = await pineconeIndex.query({
  queryRequest: {
    vector: queryEmbedding.values,
    includeMetadata: true,
    includeValues: true,
    namespace: "default",
    filter: {
      section: { "$eq": "Sports" }
    },
    topK: 1000
  }
});

// We extract the vectors of the results
const userVectors = queryResult?.matches?.map((result: any) => result.values as number[]);

// A couple of functions to calculate mean vector
const mean = (arr: number[]): number => arr.reduce((a, b) => a + b, 0) / arr.length;
const meanVector = (vectors: number[][]): number[] => {
  const { length } = vectors[0];

  return Array.from({ length }).map((_, i) =>
    mean(vectors.map(vec => vec[i]))
  );
};

// We calculate the mean vector of the results
const meanVec = meanVector(userVectors!);

// We query the index with the mean vector to get recommendations for the user
const recommendations = await pineconeIndex.query({
  queryRequest: {
    vector: meanVec,
    includeMetadata: true,
    includeValues: true,
    namespace: "default",
    topK: 10
  }
});

// select the first 10 results

const results = new dfd.DataFrame(queryResult?.matches?.slice(0, 10).map((result: any) => result.metadata));

results.print();


const recs = new dfd.DataFrame(recommendations?.matches?.map((result: any) => result.metadata));

recs.print();
