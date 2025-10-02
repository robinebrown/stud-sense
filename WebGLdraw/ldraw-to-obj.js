#!/usr/bin/env node
"use strict";

const fs          = require("fs");
const path        = require("path");
const { Parser }  = require("ldraw-parser");
const { OBJExporter } = require("ldraw-to-obj");

// 1) Folder of all part .dat definitions
const inputDir  = path.join(__dirname, "parts/ldraw/parts");
// 2) Where to write .obj files
const outputDir = path.join(__dirname, "objs");

// Ensure output folder exists
fs.mkdirSync(outputDir, { recursive: true });

// Recursively collect every .dat under the parts directory
function walkDatFiles(dir) {
  let results = [];
  fs.readdirSync(dir).forEach(name => {
    const full = path.join(dir, name);
    if (fs.statSync(full).isDirectory()) {
      results = results.concat(walkDatFiles(full));
    } else if (name.toLowerCase().endsWith(".dat")) {
      results.push(full);
    }
  });
  return results;
}

const datFiles = walkDatFiles(inputDir);
if (!datFiles.length) {
  console.error("✖ No .dat files found under", inputDir);
  process.exit(1);
}
console.log(`→ Found ${datFiles.length} parts to convert.`);

// For each .dat, parse & export to OBJ
const exporter = new OBJExporter();
datFiles.forEach(datPath => {
  const name = path.basename(datPath, ".dat");
  try {
    const data  = fs.readFileSync(datPath, "utf8");
    const model = new Parser().parse(data);       // resolves all dependencies/p/48
    const obj   = exporter.export(model);         // triangulates, tessellates arcs
    fs.writeFileSync(path.join(outputDir, name + ".obj"), obj);
    console.log(`✓ ${name}.obj`);
  } catch (err) {
    console.error(`✖ Failed ${name}: ${err.message}`);
  }
});
