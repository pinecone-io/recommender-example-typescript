import yargs from "yargs";
import { hideBin } from "yargs/helpers";

// Parses the `--query` and `--section` arguments used by `npm run recommend`.
// Both are required; yargs prints usage and exits if either is missing.
export const getQueryingCommandLineArguments = () => {
  const argv = yargs(hideBin(process.argv))
    .option("query", {
      alias: "q",
      type: "string",
      description: "The query to search for",
      demandOption: true,
    })
    .option("section", {
      alias: "s",
      type: "string",
      description: "The section of the article",
      demandOption: true,
    })
    .parseSync();

  const { query, section } = argv;
  return { query, section };
};
