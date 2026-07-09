import { describe, expect, it, vi } from "vitest";
import type { Index, PineconeRecord } from "@pinecone-database/pinecone";
import { chunkedUpsert } from "../src/utils/chunkedUpsert.ts";

// Builds a minimal mock of a Pinecone `Index` that records every upsert call,
// grouped by namespace, so we can assert on chunking behaviour without a
// network connection.
function makeMockIndex() {
  const upsert = vi.fn().mockResolvedValue(undefined);
  const namespace = vi.fn(() => ({ upsert }));
  const index = { namespace } as unknown as Index;
  return { index, namespace, upsert };
}

const record = (id: string): PineconeRecord => ({ id, values: [0.1, 0.2] });

describe("chunkedUpsert", () => {
  it("splits vectors into chunks of the given size", async () => {
    const { index, namespace, upsert } = makeMockIndex();
    const vectors = Array.from({ length: 25 }, (_, i) => record(`${i}`));

    const result = await chunkedUpsert(index, vectors, "ns", 10);

    expect(result).toBe(true);
    // 25 vectors, chunk size 10 => 3 upsert calls (10, 10, 5)
    expect(upsert).toHaveBeenCalledTimes(3);
    expect(upsert.mock.calls[0][0].records).toHaveLength(10);
    expect(upsert.mock.calls[1][0].records).toHaveLength(10);
    expect(upsert.mock.calls[2][0].records).toHaveLength(5);
    expect(namespace).toHaveBeenCalledWith("ns");
  });

  it("defaults to a chunk size of 10", async () => {
    const { index, upsert } = makeMockIndex();
    const vectors = Array.from({ length: 11 }, (_, i) => record(`${i}`));

    await chunkedUpsert(index, vectors, "ns");

    expect(upsert).toHaveBeenCalledTimes(2);
    expect(upsert.mock.calls[0][0].records).toHaveLength(10);
    expect(upsert.mock.calls[1][0].records).toHaveLength(1);
  });

  it("does not upsert anything for an empty vector list", async () => {
    const { index, upsert } = makeMockIndex();

    const result = await chunkedUpsert(index, [], "ns");

    expect(result).toBe(true);
    expect(upsert).not.toHaveBeenCalled();
  });

  it("swallows per-chunk errors and still resolves to true", async () => {
    const { index, upsert } = makeMockIndex();
    upsert.mockRejectedValueOnce(new Error("boom"));
    const vectors = Array.from({ length: 15 }, (_, i) => record(`${i}`));

    const result = await chunkedUpsert(index, vectors, "ns", 10);

    expect(result).toBe(true);
    expect(upsert).toHaveBeenCalledTimes(2);
  });
});
