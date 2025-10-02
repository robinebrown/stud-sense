#!/usr/bin/env node
"use strict";

const fs        = require("fs");
const path      = require("path");
const DATParser = require("./DATParser_v2");

// input .dat directory and output OBJ directory
const inputDir  = path.join(__dirname, "parts/ldraw/parts");
const outputDir = path.join(__dirname, "objs_v2");

if (!fs.existsSync(inputDir)) {
  console.error("✖ Input dir not found:", inputDir);
  process.exit(1);
}
fs.mkdirSync(outputDir, { recursive: true });

// recursively list .dat files
function walk(dir) {
  let out = [];
  fs.readdirSync(dir).forEach(name => {
    const full = path.join(dir, name);
    if (fs.statSync(full).isDirectory()) out = out.concat(walk(full));
    else if (name.toLowerCase().endsWith(".dat")) out.push(full);
  });
  return out;
}

const datFiles = walk(inputDir);
console.log(`Converting ${datFiles.length} parts → ${outputDir}`);

datFiles.forEach(datPath => {
  const partName = path.basename(datPath, ".dat");
  process.stdout.write(` → ${partName}… `);

  let meshes;
  try {
    meshes = new DATParser().parseFile(datPath);
  } catch (err) {
    console.error("parse error:", err.message);
    return;
  }

  let obj;
  try {
    obj = new DATParser().toOBJ(meshes);
  } catch (err) {
    console.error("toOBJ error:", err.message);
    return;
  }

  fs.writeFileSync(
    path.join(outputDir, partName + ".obj"),
    obj,
    "utf8"
  );
  console.log("done");
});

console.log("All done — check objs_v2/");
