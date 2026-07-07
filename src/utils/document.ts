// Minimal stand-in for langchain's `Document`. This project only ever used
// langchain for this tiny value type, so we inline it here and drop the
// dependency (which pulled in a large, vulnerable transitive tree).
//
// The metadata default is `Record<string, any>` to match langchain's original
// signature: it must stay assignable to Pinecone's `RecordMetadata` at the call
// sites in embeddings.ts.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface DocumentInput<Metadata = Record<string, any>> {
  pageContent: string;
  metadata?: Metadata;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export class Document<Metadata = Record<string, any>> {
  pageContent: string;

  metadata: Metadata;

  constructor(fields: DocumentInput<Metadata>) {
    this.pageContent = fields.pageContent;
    this.metadata = fields.metadata ?? ({} as Metadata);
  }
}
