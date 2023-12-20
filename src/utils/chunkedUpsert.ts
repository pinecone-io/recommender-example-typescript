import type { Index, PineconeRecord } from "@pinecone-database/pinecone";
import { sliceIntoChunks } from "./util.ts";

export const chunkedUpsert = async (
  index: Index,
  vectors: Array<PineconeRecord>,
  namespace: string,
  chunkSize = 10
) => {
  // Split the vectors into chunks
  const chunks = sliceIntoChunks<PineconeRecord>(vectors, chunkSize);

  try {
    // Upsert each chunk of vectors into the index
    await Promise.allSettled(
      chunks.map(async (chunk) => {
        try {
          await index.namespace(namespace).upsert(chunk);
        } catch (e) {
          console.log("Error upserting chunk", e);
        }
      })
    );

    return true;
  } catch (e) {
    throw new Error(`Error upserting vectors into index: ${e}`);
  }
};
