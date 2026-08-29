// Exercises a real upsert/query round-trip against a live Pinecone project —
// the gap that let a v2→v8 response-shape mismatch (see #5) ship undetected,
// since unit tests mock the SDK entirely.
//
// Skips outright when PINECONE_API_KEY is unset so `npm test` and PR runs
// stay green and credential-free; only `npm run test:integration` (wired into
// CI on main/workflow_dispatch, see ci.yml) actually exercises it.
import { randomUUID } from "crypto";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  Pinecone,
  type Index,
  type PineconeRecord,
} from "@pinecone-database/pinecone";
import { getEnv } from "../../src/utils/env.ts";
import { chunkedUpsert } from "../../src/utils/chunkedUpsert.ts";

const DIMENSION = 384;
const NAMESPACE = "integration-test";

const randomVector = (length: number): number[] =>
  Array.from({ length }, () => Math.random());

// Serverless upserts are eventually consistent, so a query issued right after
// an upsert can legitimately return fewer than the expected matches. Poll
// instead of a fixed sleep.
async function waitForMatches(
  index: Index,
  vector: number[],
  expected: number,
  { attempts = 10, delayMs = 3000 } = {}
) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const result = await index.namespace(NAMESPACE).query({
      vector,
      topK: expected,
      includeMetadata: true,
      includeValues: true,
    });
    if ((result.matches?.length ?? 0) >= expected) {
      return result;
    }
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
  throw new Error(`Query never returned ${expected} match(es) in time`);
}

describe.skipIf(!process.env.PINECONE_API_KEY)("Pinecone integration", () => {
  const indexName = `recommender-it-${randomUUID().slice(0, 8)}`;
  let pinecone: Pinecone;
  let index: Index;
  let indexCreated = false;

  beforeAll(async () => {
    pinecone = new Pinecone({ apiKey: getEnv("PINECONE_API_KEY") });
    await pinecone.createIndex({
      name: indexName,
      dimension: DIMENSION,
      metric: "cosine",
      spec: {
        serverless: {
          cloud: getEnv("PINECONE_CLOUD"),
          region: getEnv("PINECONE_REGION"),
        },
      },
      waitUntilReady: true,
    });
    indexCreated = true;
    index = pinecone.index(indexName);
  }, 120_000);

  afterAll(async () => {
    if (indexCreated) {
      await pinecone.deleteIndex(indexName);
    }
  });

  it("upserts and queries records through the real chunkedUpsert path", async () => {
    const records: PineconeRecord[] = Array.from({ length: 5 }, (_, i) => ({
      id: `article-${i}`,
      values: randomVector(DIMENSION),
      metadata: { title: `Article ${i}` },
    }));

    await chunkedUpsert(index, records, NAMESPACE, 2);

    const result = await waitForMatches(
      index,
      records[0].values as number[],
      records.length
    );

    expect(result.matches?.length).toBeGreaterThan(0);
    for (const match of result.matches ?? []) {
      expect(match.id).toMatch(/^article-\d+$/);
      expect(match.values).toHaveLength(DIMENSION);
      expect(match.metadata?.title).toBeDefined();
    }
  }, 60_000);
});
