import fs from "fs/promises";
import * as dfd from "danfojs-node";

async function loadCSVFile(
  filePath: string
): Promise<dfd.DataFrame> {
  try {
    // Get csv file absolute path
    const csvAbsolutePath = await fs.realpath(filePath);

    // try {
    //   // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    //   // @ts-ignore
    //   fs.writeFile(filePath, csvAbsolutePath, (err) => {
    //     if (err) throw err;
    //   });
    // } catch (err) {
    //   console.log(err);
    // }

    const df: dfd.DataFrame = (await dfd.readCSV(csvAbsolutePath)) as dfd.DataFrame;

    // Create a readable stream from the CSV file
    return df;
  } catch (err) {
    console.error(err);
    throw err;
  }
}

export default loadCSVFile;
