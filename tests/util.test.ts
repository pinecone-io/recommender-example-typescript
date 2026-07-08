import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { getEnv, sliceIntoChunks } from "../src/utils/util.ts";

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

describe("getEnv", () => {
  const KEY = "RECOMMENDER_TEST_ENV_VAR";
  let original: string | undefined;

  beforeEach(() => {
    original = process.env[KEY];
    delete process.env[KEY];
  });

  afterEach(() => {
    if (original === undefined) {
      delete process.env[KEY];
    } else {
      process.env[KEY] = original;
    }
  });

  it("returns the value when the variable is set", () => {
    process.env[KEY] = "hello";
    expect(getEnv(KEY)).toBe("hello");
  });

  it("throws a descriptive error when the variable is missing", () => {
    expect(() => getEnv(KEY)).toThrowError(
      `${KEY} environment variable not set`
    );
  });

  it("throws when the variable is set but empty", () => {
    process.env[KEY] = "";
    expect(() => getEnv(KEY)).toThrowError();
  });
});
