import { randomUUID } from "crypto";
import { Pipeline, pipeline, AutoConfig } from "@xenova/transformers";
import type { PineconeRecord, RecordMetadata } from "@pinecone-database/pinecone";
import type { Document } from 'langchain/document';
import { EmbeddingsParams, Embeddings } from "langchain/embeddings/base";
import { sliceIntoChunks } from "./utils/util.js";


type DocumentOrString = Document | string;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isString(test: any): test is string {
  return typeof test === "string";
}

class Embedder {
  private pipe: Pipeline;

  async init(modelName: string) {
    const config = await AutoConfig.from_pretrained(modelName);
    this.pipe = await pipeline(
      "embeddings",
      modelName,
      {
        quantized: false,
        config
      }
    );
  }

  // Embeds a text and returns the embedding
  async embed(text: string, metadata?: RecordMetadata): Promise<PineconeRecord> {
    try {
      const result = await this.pipe(text, { pooling: 'mean', normalize: true });
      const id = (metadata?.id as string) || randomUUID();
      return {
        id,
        metadata: metadata || {
          text,
        },
        values: Array.from(result.data) as number[],
      };
    } catch (e) {
      console.log(`Error embedding text: ${text}, ${e}`);
      throw e;
    }
  }

  // Embeds a batch of documents and calls onDoneBatch with the embeddings
  async embedBatch(
    documents: DocumentOrString[],
    batchSize: number,
    onDoneBatch: (embeddings: PineconeRecord[]) => void
  ) {
    const batches = sliceIntoChunks<DocumentOrString>(documents, batchSize);
    for (const batch of batches) {
      const embeddings = await Promise.all(
        batch.map((documentOrString) =>
          isString(documentOrString)
            ? this.embed(documentOrString)
            : this.embed(documentOrString.pageContent, documentOrString.metadata)
        )
      );
      await onDoneBatch(embeddings);
    }
  }
}

interface TransformersJSEmbeddingParams extends EmbeddingsParams {
  modelName: string;
  onEmbeddingDone?: (embeddings: PineconeRecord[]) => void;
}

class TransformersJSEmbedding extends Embeddings implements TransformersJSEmbeddingParams {
  modelName: string;

  pipe: Pipeline | null = null;

  constructor(params: TransformersJSEmbeddingParams) {
    super(params);
    this.modelName = params.modelName;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    this.pipe = this.pipe || await pipeline(
      "embeddings",
      this.modelName
    );

    const embeddings = await Promise.all(texts.map(async (text) => this.embedQuery(text)));
    return embeddings;
  }

  async embedQuery(text: string): Promise<number[]> {
    this.pipe = this.pipe || await pipeline(
      "embeddings",
      this.modelName
    );

    const result = await this.pipe(text);
    return Array.from(result.data) as number[];
  }
}


const embedder = new Embedder();
export { embedder, TransformersJSEmbedding };
