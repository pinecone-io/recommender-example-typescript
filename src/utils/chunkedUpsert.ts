import type { Index, PineconeRecord } from "@pinecone-database/pinecone";
import { sliceIntoChunks } from "./chunk.ts";

// Upserts vectors into an index namespace in batches of `chunkSize`. Batches run
// concurrently; a failed batch is logged and skipped so a single bad chunk does
// not abort the whole ingestion run.
export const chunkedUpsert = async (
  index: Index,
  vectors: PineconeRecord[],
  namespace: string,
  chunkSize = 10
) => {
  const chunks = sliceIntoChunks<PineconeRecord>(vectors, chunkSize);

  await Promise.all(
    chunks.map(async (chunk) => {
      try {
        await index.namespace(namespace).upsert({ records: chunk });
      } catch (e) {
        console.error("Error upserting chunk", e);
      }
    })
  );
};
