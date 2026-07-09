import fs from "fs/promises";
import { parseCSV, type CSVRow } from "./csv.ts";

async function loadCSVFile(filePath: string): Promise<CSVRow[]> {
  try {
    // Get csv file absolute path
    const csvAbsolutePath = await fs.realpath(filePath);
    const content = await fs.readFile(csvAbsolutePath, "utf-8");
    return parseCSV(content);
  } catch (err) {
    console.error(err);
    throw err;
  }
}

export default loadCSVFile;
