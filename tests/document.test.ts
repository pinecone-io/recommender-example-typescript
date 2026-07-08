import { describe, expect, it } from "vitest";
import { Document } from "../src/utils/document.ts";

// Guards the inlined stand-in for langchain's `Document` (dropped when the
// langchain dependency was removed). If these break, the embeddings pipeline
// that consumes `pageContent` / `metadata` is at risk.
describe("Document", () => {
  it("stores pageContent and metadata as provided", () => {
    const doc = new Document({
      pageContent: "some text",
      metadata: { id: "1", section: "intro" },
    });
    expect(doc.pageContent).toBe("some text");
    expect(doc.metadata).toEqual({ id: "1", section: "intro" });
  });

  it("defaults metadata to an empty object when omitted", () => {
    const doc = new Document({ pageContent: "some text" });
    expect(doc.metadata).toEqual({});
  });

  it("preserves an explicitly empty pageContent", () => {
    const doc = new Document({ pageContent: "" });
    expect(doc.pageContent).toBe("");
  });
});
