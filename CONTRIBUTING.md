# Contributing

Thanks for your interest in improving this example! It demonstrates a
content-based article recommender built on Pinecone similarity search, and
contributions that keep it clear, correct, and up to date are very welcome.

## Prerequisites

- **Node.js 20 or newer** (CI runs the suite on Node 20 and 22).
- A Pinecone API key from the [Pinecone console](https://app.pinecone.io) if you
  want to run the demo end to end.

## Getting started

```bash
npm install
```

To run the demo scripts you also need Pinecone credentials. Copy the template
and fill in your values:

```bash
cp .env.example .env
```

See the [README](./README.md) for what each variable means and how to download
the dataset.

## Running the checks

Please run the same checks CI does before opening a pull request:

```bash
npm run lint          # ESLint
npm run format:check  # Prettier (use `npm run format` to auto-fix)
npm run typecheck     # tsc --noEmit
npm test              # Vitest (unit tests)
```

`npm run test:watch` is handy while iterating on tests.

## Running the demo

Once your `.env` is configured and the dataset is in place (see the README):

```bash
npm run index      # embed the dataset and upsert into Pinecone
npm run recommend  # query for recommendations
```

## Opening a pull request

1. Fork the repo and create a topic branch (`docs/…`, `fix/…`, `chore/…`).
2. Make a focused change and keep the diff small.
3. Ensure all the checks above pass.
4. Open a pull request describing **what** changed and **why**.

By contributing, you agree that your contributions are licensed under the
project's [MIT License](./LICENSE) and that you will follow our
[Code of Conduct](./CODE_OF_CONDUCT.md).
