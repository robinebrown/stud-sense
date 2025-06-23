#!/usr/bin/env node
"use strict";

const fs   = require("fs");
const path = require("path");
const DATParser = require("./DATParser");

// Grab CLI args
const [ inputDir, outputDir ] = process.argv.slice(2);

// Debug: confirm we’re seeing the right folder
console.log("convert.js started");
console.log("→ inputDir:", inputDir, "exists?", fs.existsSync(inputDir));

if (!inputDir || !outputDir) {
  console.error("Usage: node convert.js <input_dat_folder> <output_obj_folder>");
  process.exit(1);
}

/**
 * Recursively collect all .dat files under the given directory
 */
function walkDatFiles(dir) {
  let results = [];
  fs.readdirSync(dir).forEach(name => {
    const full = path.join(dir, name);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      results = results.concat(walkDatFiles(full));
    } else if (name.toLowerCase().endsWith('.dat')) {
      results.push(full);
    }
  });
  return results;
}

// Find all .dat files
const datFiles = walkDatFiles(inputDir);
console.log(`→ found ${datFiles.length} .dat files under ${inputDir}`);
if (datFiles.length < 20) console.log("→ files:", datFiles);

// Ensure output folder exists
fs.mkdirSync(outputDir, { recursive: true });

// Convert each .dat to .obj
datFiles.forEach(datPath => {
  const filename = path.basename(datPath);
  console.log(`→ Converting ${filename} ...`);

  // 1) Parse (will recurse into includes via DATParser)
  let meshes;
  try {
    meshes = DATParser.parseFile(datPath);
  } catch (err) {
    console.error(`✖ Failed to load dependencies for ${filename}: ${err.message}`);
    return;
  }

  // 2) Convert to OBJ text
  let objText;
  try {
    objText = DATParser.toOBJ(meshes);
  } catch (err) {
    console.error(`✖ Failed to convert ${filename}: ${err.message}`);
    return;
  }

  // 3) Write out the OBJ file
  try {
    const base    = path.basename(filename, path.extname(filename));
    const outPath = path.join(outputDir, base + ".obj");
    fs.writeFileSync(outPath, objText, "utf8");
  } catch (err) {
    console.error(`✖ Could not write OBJ for ${filename}: ${err.message}`);
  }
});

console.log("All done.");
