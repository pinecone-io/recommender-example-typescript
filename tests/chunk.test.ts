import { describe, it, expect } from "vitest";
import { sliceIntoChunks } from "../src/utils/chunk.ts";

describe("sliceIntoChunks", () => {
  it("splits an array into evenly sized chunks", () => {
    expect(sliceIntoChunks([1, 2, 3, 4], 2)).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it("puts the remainder in a final, smaller chunk", () => {
    expect(sliceIntoChunks([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
  });

  it("returns a single chunk when the chunk size exceeds the length", () => {
    expect(sliceIntoChunks([1, 2, 3], 10)).toEqual([[1, 2, 3]]);
  });

  it("returns an empty array for empty input", () => {
    expect(sliceIntoChunks([], 3)).toEqual([]);
  });
});
