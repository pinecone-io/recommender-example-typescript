import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { getEnv } from "../src/utils/env.ts";

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
