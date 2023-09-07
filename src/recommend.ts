/* eslint-disable import/no-extraneous-dependencies */
import { getEnv, getQueryingCommandLineArguments } from "utils/util.ts";
import { embedder } from "embeddings.ts";
import { Table } from 'console-table-printer';
import { Pinecone } from "@pinecone-database/pinecone";
import type { ScoredPineconeRecord } from "@pinecone-database/pinecone";
import type { ArticleRecord } from "types.ts";

const indexName = getEnv("PINECONE_INDEX");
const pinecone = new Pinecone();

// Ensure the index exists
try {
  const description = await pinecone.describeIndex(indexName);
  if (!description.status?.ready) {
    throw `Index not ready, description was ${JSON.stringify(description)}`
  }
} catch (e) {
  console.log('An error occurred. Run "npm run index" to load data into the index before querying.')
  throw e;
}

const index = pinecone.index<ArticleRecord>(indexName).namespace('default');

await embedder.init("Xenova/all-MiniLM-L6-v2");

const { query, section } = getQueryingCommandLineArguments();

// We create a simulated user with an interest given a query and a specific section
const queryEmbedding = await embedder.embed(query)
const queryResult = await index.query({
    vector: queryEmbedding.values,
    includeMetadata: true,
    includeValues: true,
    filter: {
      section: { "$eq": section }
    },
    topK: 10
});

// We extract the vectors of the results
const userVectors = queryResult?.matches?.map((result: ScoredPineconeRecord<ArticleRecord>) => result.values);

// A couple of functions to calculate mean vector
const mean = (arr: number[]): number => arr.reduce((a, b) => a + b, 0) / arr.length;
const meanVector = (vectors: number[][]): number[] => {
  const { length } = vectors[0];

  return Array.from({ length }).map((_, i) =>
    mean(vectors.map(vec => vec[i]))
  );
};

// We calculate the mean vector of the results
// eslint-disable-next-line @typescript-eslint/no-non-null-assertion
const meanVec = meanVector(userVectors!);

// We query the index with the mean vector to get recommendations for the user
const recommendations = await index.query({
    vector: meanVec,
    includeMetadata: true,
    includeValues: true,
    topK: 10
});

const userPreferences = new Table({
  columns: [
    { name: "title", alignment: "left" },
    { name: "author", alignment: "left" },
    { name: "section", alignment: "left" },
  ]
});

const userRecommendations = new Table({
  columns: [
    { name: "title", alignment: "left" },
    { name: "author", alignment: "left" },
    { name: "section", alignment: "left" },
  ]
});

queryResult?.matches?.slice(0, 10).forEach((result: any) => {
  const { title, article, publication, section } = result.metadata;
  userPreferences.addRow({
    title,
    article: `${article.slice(0, 70)}...`,
    publication,
    section
  });
});

console.log("========== User Preferences ==========");
userPreferences.printTable();

recommendations?.matches?.slice(0, 10).forEach((result: any) => {
  const { title, article, publication, section } = result.metadata;
  userRecommendations.addRow({
    title,
    article: `${article.slice(0, 70)}...`,
    publication,
    section
  });
});
console.log("=========== Recommendations ==========");
userRecommendations.printTable();
