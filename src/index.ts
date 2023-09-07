/* eslint-disable import/no-extraneous-dependencies */
/* eslint-disable dot-notation */
import * as dotenv from "dotenv";
import { Pinecone, type PineconeRecord } from '@pinecone-database/pinecone';
import { getEnv } from "utils/util.ts";
import cliProgress from "cli-progress";
import { Document } from 'langchain/document';
import * as dfd from "danfojs-node";
import { embedder } from "embeddings.ts";
import loadCSVFile from "utils/csvLoader.ts";
import splitFile from "utils/fileSplitter.ts";
import { chunkedUpsert } from './utils/chunkedUpsert.ts';

type ArticleRecord = {
  index: number,
  title: string;
  article: string;
  publication: string;
  url: string;
  author: string;
  section: string;
}

dotenv.config();

const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);

// Index setup
const indexName = getEnv("PINECONE_INDEX");
const pinecone = new Pinecone();

async function getChunk(df: dfd.DataFrame, start: number, size: number): Promise<dfd.DataFrame> {
  // eslint-disable-next-line no-return-await
  return await df.head(start + size).tail(size);
}

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

async function embedAndUpsert(dataFrame: dfd.DataFrame, chunkSize: number) {
  const chunkGenerator = processInChunks<ArticleRecord, 'section' | 'url' | 'title' | 'publication' | 'author' | 'article', 'article'>(
    dataFrame,
    100,
    ['section', 'url', 'title', 'publication', 'author', 'article'],
    'article'
  );
  const index = pinecone.index(indexName);

  for await (const documents of chunkGenerator) {
    await embedder.embedBatch(documents, chunkSize, async (embeddings: PineconeRecord[]) => {
      await chunkedUpsert(index, embeddings, "default");
      progressBar.increment(embeddings.length);
    });
  }
}

try {
  const fileParts = await splitFile("./data/all-the-news-2-1.csv", 500000);
  const firstFile = fileParts[0];

  // For this example, we will use the first file part to create the index
  const data = await loadCSVFile(firstFile);
  const clean = data.dropNa() as dfd.DataFrame;
  clean.head().print();

  // Create the index if it doesn't already exist
  const indexList = await pinecone.listIndexes();
  if (indexList.indexOf({ name: indexName }) === -1) {
    await pinecone.createIndex({ name: indexName, dimension: 384, waitUntilReady: true })
  }

  progressBar.start(clean.shape[0], 0);
  await embedder.init("Xenova/all-MiniLM-L6-v2");
  await embedAndUpsert(clean, 1);
  progressBar.stop();
  console.log(`Inserted ${progressBar.getTotal()} documents into index ${indexName}`);

} catch (error) {
  console.error(error);
}
