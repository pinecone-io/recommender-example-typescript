import { TransformersJSEmbedding } from '../embeddings';

// Mock the pipeline function from @xenova/transformers
jest.mock('@xenova/transformers', () => ({
  pipeline: jest.fn().mockImplementation(() => {
    // Return a function that simulates the behavior of this.pipe
    return async () => {
      return { data: new Float32Array([1, 2, 3, 4]) };  // Mocked data
    };
  }),
}));

describe('TransformersJSEmbedding', () => {
  test('should return embeddings for a given text', async () => {
    const transformer = new TransformersJSEmbedding({ modelName: 'some-model' });
    const result = await transformer.embedQuery('some text');
    expect(result).toEqual([1, 2, 3, 4]);  // Adjust this to match your expected output
  });
});
