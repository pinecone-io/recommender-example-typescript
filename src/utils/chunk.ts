// Splits an array into consecutive chunks of at most `chunkSize` items. Used
// to batch documents for embedding and vectors for upserting.
export const sliceIntoChunks = <T>(arr: T[], chunkSize: number): T[][] =>
  Array.from({ length: Math.ceil(arr.length / chunkSize) }, (_, i) =>
    arr.slice(i * chunkSize, (i + 1) * chunkSize)
  );
