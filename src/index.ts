/* eslint-disable import/no-extraneous-dependencies */
/* eslint-disable dot-notation */
import * as dotenv from "dotenv";
import {
  Pinecone,
  type PineconeRecord,
  type ServerlessSpecCloudEnum,
} from "@pinecone-database/pinecone";
import { getEnv, validateEnvironmentVariables } from "utils/util.ts";
import cliProgress from "cli-progress";
import { embedder } from "embeddings.ts";
import loadCSVFile from "utils/csvLoader.ts";
import { dropEmptyRows } from "utils/csv.ts";
import splitFile from "utils/fileSplitter.ts";
import type { ArticleRecord } from "types.ts";
import { Document } from "./utils/document.ts";
import { chunkedUpsert } from "./utils/chunkedUpsert.ts";

dotenv.config();
validateEnvironmentVariables();

const progressBar = new cliProgress.SingleBar(
  {},
  cliProgress.Presets.shades_classic
);

// Index setup
const indexName = getEnv("PINECONE_INDEX");
const indexCloud = getEnv("PINECONE_CLOUD") as ServerlessSpecCloudEnum;
const indexRegion = getEnv("PINECONE_REGION");
const pinecone = new Pinecone();

async function* processInChunks<T, M extends keyof T, P extends keyof T>(
  records: T[],
  chunkSize: number,
  metadataFields: M[],
  pageContentField: P
): AsyncGenerator<Document[]> {
  for (let i = 0; i < records.length; i += chunkSize) {
    const chunk = records.slice(i, i + chunkSize);
    yield chunk.map((record: T) => {
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

async function embedAndUpsert(records: ArticleRecord[], chunkSize: number) {
  const chunkGenerator = processInChunks<
    ArticleRecord,
    "section" | "url" | "title" | "publication" | "author" | "article",
    "article"
  >(
    records,
    100,
    ["section", "url", "title", "publication", "author", "article"],
    "article"
  );
  const index = pinecone.index(indexName);

  for await (const documents of chunkGenerator) {
    await embedder.embedBatch(
      documents,
      chunkSize,
      async (embeddings: PineconeRecord[]) => {
        await chunkedUpsert(index, embeddings, "default");
        progressBar.increment(embeddings.length);
      }
    );
  }
}

try {
  const fileParts = await splitFile("./data/all-the-news-2-1.csv", 100000);
  const firstFile = fileParts[0];

  // For this example, we will use the first file part to create the index
  const data = await loadCSVFile(firstFile);
  const clean = dropEmptyRows(data);
  console.table(clean.slice(0, 5));

  // Create the index if it doesn't already exist
  const indexList = await pinecone.listIndexes();
  if (!indexList.indexes?.some((index) => index.name === indexName)) {
    await pinecone.createIndex({
      name: indexName,
      dimension: 384,
      spec: { serverless: { region: indexRegion, cloud: indexCloud } },
      waitUntilReady: true,
    });
  }

  progressBar.start(clean.length, 0);
  await embedder.init("Xenova/all-MiniLM-L6-v2");
  await embedAndUpsert(clean as unknown as ArticleRecord[], 1);
  progressBar.stop();
  console.log(
    `Inserted ${progressBar.getTotal()} documents into index ${indexName}`
  );
} catch (error) {
  console.error(error);
}
