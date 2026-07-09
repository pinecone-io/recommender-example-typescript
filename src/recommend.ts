// Queries the article index for a simulated user.
//
// Given a `--query` and `--section`, we fetch the articles the user has
// "read" (the query results), average their vectors into a single taste
// vector, then query again with that mean to produce recommendations.
//
// Run with: npm run recommend -- --query="tennis" --section="Sports"
import { Pinecone } from "@pinecone-database/pinecone";
import type { ScoredPineconeRecord } from "@pinecone-database/pinecone";
import { Table } from "console-table-printer";
import { getEnv, validateEnvironmentVariables } from "./utils/env.ts";
import { getQueryingCommandLineArguments } from "./utils/cli.ts";
import { embedder } from "./embeddings.ts";
import { DEFAULT_NAMESPACE } from "./constants.ts";
import type { ArticleRecord } from "./types.ts";

// Averages a set of equal-length vectors component by component. The resulting
// mean vector represents the user's interests across the articles they've read.
const meanVector = (vectors: number[][]): number[] => {
  const mean = (values: number[]): number =>
    values.reduce((a, b) => a + b, 0) / values.length;
  const { length } = vectors[0];

  return Array.from({ length }).map((_, i) => mean(vectors.map((v) => v[i])));
};

// Prints a table of article matches under the given heading.
const printArticleTable = (
  heading: string,
  matches: ScoredPineconeRecord<ArticleRecord>[]
) => {
  const table = new Table({
    columns: [
      { name: "title", alignment: "left" },
      { name: "article", alignment: "left" },
      { name: "section", alignment: "left" },
      { name: "publication", alignment: "left" },
    ],
  });

  for (const match of matches) {
    const { metadata } = match;
    if (metadata) {
      const { title, article, section, publication } = metadata;
      table.addRow({
        title,
        article: `${article.slice(0, 70)}...`,
        section,
        publication,
      });
    }
  }

  console.log(heading);
  table.printTable();
};

async function main() {
  validateEnvironmentVariables();

  const indexName = getEnv("PINECONE_INDEX");
  const pinecone = new Pinecone();

  // Ensure the index exists and is ready before we query it.
  try {
    const description = await pinecone.describeIndex(indexName);
    if (!description.status?.ready) {
      throw new Error(
        `Index not ready, description was ${JSON.stringify(description)}`
      );
    }
  } catch (e) {
    console.log(
      'An error occurred. Run "npm run index" to load data into the index before querying.'
    );
    throw e;
  }

  const index = pinecone
    .index<ArticleRecord>(indexName)
    .namespace(DEFAULT_NAMESPACE);

  await embedder.init("Xenova/all-MiniLM-L6-v2");

  const { query, section } = getQueryingCommandLineArguments();

  // Simulate the articles the user has read: the closest matches to their query
  // within the section they're interested in.
  const queryEmbedding = await embedder.embed(query);
  const queryResult = await index.query({
    vector: queryEmbedding.values ?? [],
    includeMetadata: true,
    includeValues: true,
    filter: { section: { $eq: section } },
    topK: 10,
  });

  const readArticles = queryResult.matches ?? [];
  if (readArticles.length === 0) {
    console.log(
      `No articles found for section "${section}" matching "${query}". ` +
        "Try a different --query or --section."
    );
    return;
  }

  // Average the read articles' vectors into a single "taste" vector.
  const userVectors = readArticles
    .map((match) => match.values)
    .filter((values): values is number[] => values !== undefined);
  const meanVec = meanVector(userVectors);

  // Query again with the taste vector to produce recommendations.
  const recommendations = await index.query({
    vector: meanVec,
    includeMetadata: true,
    includeValues: true,
    topK: 10,
  });

  printArticleTable("========== User Preferences ==========", readArticles);
  printArticleTable(
    "=========== Recommendations ==========",
    recommendations.matches ?? []
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
