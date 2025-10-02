// DATParser_v2.js
"use strict";
const fs   = require("fs");
const path = require("path");

// 1) Build index of all .dat files for quick lookup
const datIndex = {};
(function buildIndex() {
  const partsRoot = path.join(__dirname, "parts");
  function walk(dir) {
    fs.readdirSync(dir).forEach(name => {
      const full = path.join(dir, name);
      if (fs.statSync(full).isDirectory()) return walk(full);
      if (name.toLowerCase().endsWith(".dat")) {
        datIndex[name.toLowerCase()] = full;
      }
    });
  }
  if (fs.existsSync(partsRoot)) walk(partsRoot);
})();

// 2) Utility to load a .dat by name or path
function getDATFile(fn) {
  let norm = fn.replace(/^['"]+|['"]+$/g, "").replace(/\\/g, "/");
  let base = path.basename(norm).toLowerCase();
  if (datIndex[base]) return fs.readFileSync(datIndex[base], "utf8");
  if (fs.existsSync(norm)) return fs.readFileSync(norm, "utf8");
  throw new Error("DATParser: file not found: " + fn);
}

// 3) Helper to parse floats
function castFloat(x) { return parseFloat(x); }

// 4) Mesh builder
function buildMesh() {
  return { vertices: [], faces: [] };
}

// 5) The Parser class
class DATParser {
  parseInstruction(line, meshes) {
    if (!line.trim()) return;
    const ops = line.trim().split(/\s+/);
    const code = ops.shift();
    const cur  = meshes[meshes.length - 1];

    if (code === "0") return;                  // comment
    if (code === "1") return this.handleDependency(ops, meshes);
    if (code === "2") return this.parseArc(ops, cur);
    if (code === "3") return this.parseTriangle(ops, cur);
    if (code === "4") return this.parseQuad(ops, cur);
  }

  handleDependency(ops, meshes) {
    // ops = [ color, ox,oy,oz, m0..m8, filename ]
    const color    = parseInt(ops[0], 10);
    const offset   = ops.slice(1, 4).map(castFloat);
    const mat      = ops.slice(4, 13).map(castFloat);
    const filename = ops[13];
    const child    = new DATParser();
    const submeshes = child.parseFile(filename);
    submeshes.forEach(mesh => {
      mesh.vertices.forEach(v => {
        const x = mat[0]*v.x + mat[1]*v.y + mat[2]*v.z + offset[0];
        const y = mat[3]*v.x + mat[4]*v.y + mat[5]*v.z + offset[1];
        const z = mat[6]*v.x + mat[7]*v.y + mat[8]*v.z + offset[2];
        v.x = x; v.y = y; v.z = z; v.color = color;
      });
      meshes.push(mesh);
    });
  }

  parseTriangle(ops, mesh) {
    // ops = [color, x1,y1,z1, x2,y2,z2, x3,y3,z3]
    const color = parseInt(ops[0], 10);
    const coords = ops.map(castFloat);
    // sample three points
    const pts = [
      coords.slice(1,4),
      coords.slice(4,7),
      coords.slice(7,10),
    ];
    // add vertices & record indices (1-based)
    const idx = pts.map(p => {
      mesh.vertices.push({ x:p[0], y:p[1], z:p[2], color });
      return mesh.vertices.length;
    });
    // add face
    mesh.faces.push({ v1: idx[0], v2: idx[1], v3: idx[2] });
  }

  parseQuad(ops, mesh) {
    // ops = [color, x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4]
    const color = parseInt(ops[0], 10);
    const coords = ops.map(castFloat);
    const pts = [
      coords.slice(1,4),
      coords.slice(4,7),
      coords.slice(7,10),
      coords.slice(10,13),
    ];
    // add vertices
    const idx = pts.map(p => {
      mesh.vertices.push({ x:p[0], y:p[1], z:p[2], color });
      return mesh.vertices.length;
    });
    // split into two tris
    mesh.faces.push({ v1: idx[0], v2: idx[1], v3: idx[2] });
    mesh.faces.push({ v1: idx[0], v2: idx[2], v3: idx[3] });
  }

  parseArc(ops, mesh) {
    // ops = [color, x0,y0,z0, x1,y1,z1, x2,y2,z2]
    const color = parseInt(ops[0], 10);
    const p0 = { x:+ops[1], y:+ops[2], z:+ops[3] };
    const p1 = { x:+ops[4], y:+ops[5], z:+ops[6] };
    const p2 = { x:+ops[7], y:+ops[8], z:+ops[9] };
    const segs = 32;
    // sample
    const idxs = [];
    for (let i = 0; i <= segs; i++) {
      const t = i/segs;
      const x = (1-t)*(1-t)*p0.x + 2*(1-t)*t*p1.x + t*t*p2.x;
      const y = (1-t)*(1-t)*p0.y + 2*(1-t)*t*p1.y + t*t*p2.y;
      const z = (1-t)*(1-t)*p0.z + 2*(1-t)*t*p1.z + t*t*p2.z;
      mesh.vertices.push({ x,y,z, color });
      idxs.push(mesh.vertices.length);
    }
    // fan‐triangulate
    for (let j = 1; j < idxs.length-1; j++) {
      mesh.faces.push({
        v1: idxs[0],
        v2: idxs[j],
        v3: idxs[j+1]
      });
    }
  }

  parseFile(filename) {
    const text = getDATFile(filename);
    const lines = text.split(/\r?\n/);
    const meshes = [ buildMesh() ];
    lines.forEach(l => this.parseInstruction(l, meshes));
    return meshes;
  }

  toOBJ(meshes, scale=0.05) {
    const nl = "\n";
    const verts = [];
    const faces = [];
    let offset = 0;
    // collect
    meshes.forEach(mesh => {
      mesh.vertices.forEach(v => {
        verts.push(
          `v ${(v.x*scale).toFixed(6)} ${(v.y*scale).toFixed(6)} ${(v.z*scale).toFixed(6)}`
        );
      });
      mesh.faces.forEach(f => {
        // f.v1 etc are 1-based within each mesh → add offset
        faces.push(
          `f ${f.v1+offset} ${f.v2+offset} ${f.v3+offset}`
        );
      });
      offset += mesh.vertices.length;
    });
    return ["# OBJ from DATParser_v4"]
      .concat(verts)
      .concat(faces)
      .join(nl);
  }
}

module.exports = DATParser;
