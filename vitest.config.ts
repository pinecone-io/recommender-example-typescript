import { configDefaults, defineConfig } from "vitest/config";

// Integration tests live under tests/integration and run via the separate
// vitest.integration.config.ts / `npm run test:integration`, never here — they
// hit a real Pinecone project and must not run on untrusted PRs.
export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    exclude: [...configDefaults.exclude, "tests/integration/**"],
    environment: "node",
  },
});
