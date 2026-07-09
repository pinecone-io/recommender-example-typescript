// Helpers for reading and validating the environment variables this example
// needs. Values are loaded from a `.env` file via `dotenv` (see the entry
// scripts) before these run.

// Reads a required environment variable, throwing a clear error if it is unset
// or empty rather than letting an `undefined` propagate deeper into the code.
export const getEnv = (key: string): string => {
  const value = process.env[key];
  if (!value) {
    throw new Error(`${key} environment variable not set`);
  }
  return value;
};

// Fails fast at startup if any required Pinecone configuration is missing.
export const validateEnvironmentVariables = () => {
  getEnv("PINECONE_API_KEY");
  getEnv("PINECONE_INDEX");
  getEnv("PINECONE_CLOUD");
  getEnv("PINECONE_REGION");
};
