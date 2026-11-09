import { defineConfig } from "vitest/config";

// Separate from vitest.config.ts so a plain `npm test` never touches the
// network: this config's tests hit a real Pinecone project and are run
// explicitly via `npm run test:integration` (see ci.yml's integration-test
// job).
export default defineConfig({
  test: {
    include: ["tests/integration/**/*.test.ts"],
    environment: "node",
    testTimeout: 120_000,
    hookTimeout: 120_000,
  },
});
