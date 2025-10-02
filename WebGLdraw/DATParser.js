// DATParser.js
var DATParser = (function () {
  "use strict";

  // ─── Node.js support for reading .dat files from disk ─────────────────
  var fs   = (typeof require === 'function') ? require('fs') : null;
  var path = (typeof require === 'function') ? require('path') : null;

  // ─── Build index of all .dat files under parts/ldraw ────────────────────
  var datIndex = {};
  if (fs && path) {
    var partsRoot = path.join(__dirname, 'parts', 'ldraw');
    function walk(dir) {
      fs.readdirSync(dir, { withFileTypes: true }).forEach(function (entry) {
        var full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          walk(full);
        } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.dat')) {
          datIndex[entry.name.toLowerCase()] = full;
        }
      });
    }
    try {
      walk(partsRoot);
    } catch (e) {
      console.warn("DATParser: couldn't walk parts/ldraw:", e.message);
    }
  }

  // ─── Cache for browser fallback ───────────────────────────────────────────
  var dataCache = {};

  // ─── Load a DAT by filename (may return empty string on missing) ──────────
  function getDATFile(filename) {
    var name = filename.replace(/\\/g, '/');
    var base = path.basename(name).toLowerCase();

    // Node.js: lookup in index
    if (fs) {
      var full = datIndex[base] || '';
      if (!full) {
        console.warn("DATParser: file not found:", filename, "(skipping)");
        return "";
      }
      return fs.readFileSync(full, 'utf8');
    }

    // Browser fallback via synchronous XHR
    if (!dataCache[filename]) {
      var xhr = new XMLHttpRequest();
      xhr.open("GET", filename, false);
      xhr.send(null);
      if (!xhr.responseText.trim()) {
        xhr.open("GET", "./parts/ldraw/parts/" + base, false);
        xhr.send(null);
        if (!xhr.responseText.trim()) {
          throw "DATParser: " + base + " was empty!";
        }
      }
      dataCache[filename] = xhr.responseText;
    }
    return dataCache[filename];
  }

  // ─── Helpers ─────────────────────────────────────────────────────────────
  function castFloat(v) { return parseFloat(v); }

  var uid = 1;
  function buildMesh() {
    return {
      meshId: uid++,
      vertices: [],
      faces: [],
      addVertex: function (x, y, z, c) {
        this.vertices.push({ x: x, y: y, z: z, color: c });
      },
      addFace: function (m, c, cw, inv) {
        var v1 = this.vertices.length; this.addVertex(m[0], m[1], m[2], c);
        var v2 = this.vertices.length; this.addVertex(m[3], m[4], m[5], c);
        var v3 = this.vertices.length; this.addVertex(m[6], m[7], m[8], c);
        var flip = (cw && inv) || (!cw && !inv);
        this.faces.push({ v1: v1, v2: flip ? v2 : v3, v3: flip ? v3 : v2 });
      },
      addQuad: function (m, c, cw, inv) {
        var v1 = this.vertices.length; this.addVertex(m[0], m[1], m[2], c);
        var v2 = this.vertices.length; this.addVertex(m[3], m[4], m[5], c);
        var v3 = this.vertices.length; this.addVertex(m[6], m[7], m[8], c);
        var v4 = this.vertices.length; this.addVertex(m[9], m[10], m[11], c);
        var flip = (cw && inv) || (!cw && !inv);
        this.faces.push({ v1: v1, v2: flip ? v2 : v3, v3: flip ? v3 : v2 });
        this.faces.push({ v1: v3, v2: flip ? v4 : v1, v3: flip ? v1 : v4 });
      },
      apply: function (c, m, color) {
        this.vertices.forEach(function (v) {
          var x = m[0]*v.x + m[1]*v.y + m[2]*v.z + c[0];
          var y = m[3]*v.x + m[4]*v.y + m[5]*v.z + c[1];
          var z = m[6]*v.x + m[7]*v.y + m[8]*v.z + c[2];
          v.x = x; v.y = y; v.z = z; v.color = color;
        });
      }
    };
  }

  // ─── Parser ──────────────────────────────────────────────────────────────
  function Parser(inverted, prefix) {
    this.invertNext = false;
    this.clockwise  = false;
    this.invert     = inverted;
    this.consolePrefix = prefix || "";
  }

  Parser.prototype = {
    log: function (s) { console.log(this.consolePrefix + s); },
    handleBFC: function (ops) {
      ops.forEach(function (op) {
        if (op === "CW")        this.clockwise = true;
        if (op === "CCW")       this.clockwise = false;
        if (op === "INVERTNEXT") this.invertNext = true;
      }, this);
    },
    handleComment: function (ops) {
      if (ops[0] === "BFC") {
        ops.shift();
        this.handleBFC(ops);
      }
    },
    handleDependency: function (ops, meshes) {
      var dep = ops.pop();
      var nums = ops.map(castFloat);
      var det = (function (m) {
        return m[0]*(m[4]*m[8] - m[7]*m[5])
             - m[3]*(m[1]*m[8] - m[7]*m[2])
             + m[6]*(m[1]*m[5] - m[4]*m[2]);
      })(nums.slice(4, 13));
      var inv = this.invertNext ? !this.invert : this.invert;
      if (det < 0) inv = !inv;
      this.invertNext = false;
      var child = new Parser(inv, this.consolePrefix);
      var childMeshes = child.parseFile(dep);
      childMeshes.forEach(function (m) {
        m.apply(nums.slice(1,4), nums.slice(4,13), nums[0]);
        meshes.push(m);
      });
    },
    parseTriangle: function (args, mesh) {
      var a = args.map(castFloat);
      mesh.addFace(a.slice(1), a[0], this.clockwise, this.invert);
    },
    parseQuad: function (args, mesh) {
      var a = args.map(castFloat);
      mesh.addQuad(a.slice(1), a[0], this.clockwise, this.invert);
    },
    parseInstruction: function (line, meshes) {
      var curr = meshes[meshes.length-1];
      if (!line.trim() && curr.vertices.length) {
        meshes.push(buildMesh());
        return;
      }
      var parts = line.trim().split(/\s+/);
      var code  = parts.shift();
      if (code === '0') this.handleComment(parts);
      else if (code === '1') this.handleDependency(parts, meshes);
      else if (code === '3') this.parseTriangle(parts, curr);
      else if (code === '4') this.parseQuad(parts, curr);
    },
    parse: function (fn, data) {
      this.log(" * parsing " + fn);
      var lines = data.split(/\r?\n/);
      var meshes = [buildMesh()];
      lines.forEach(function (ln) { this.parseInstruction(ln, meshes); }, this);
      if (!meshes[meshes.length-1].vertices.length) meshes.pop();
      return meshes;
    },
    parseFile: function (fn) {
      var txt = getDATFile(fn);
      return this.parse(fn, txt);
    },
    toOBJ: function (meshData, scale) {
      scale = scale || 0.05;
      var nl = "\n", obj = ["# LDraw⟶OBJ"], verts = ["# verts"], faces = ["# faces"], off = 1;
      meshData.forEach(function (m) {
        if (!m.vertices.length) return;
        verts.push("", "# mesh" + m.meshId);
        m.vertices.forEach(function (v) {
          verts.push(
            "v " +
            (v.x*scale).toFixed(6) + " " +
            (v.y*scale).toFixed(6) + " " +
            (v.z*scale).toFixed(6)
          );
        });
        faces.push("", "g mesh" + m.meshId, "s 1");
        m.faces.forEach(function (f) {
          faces.push(
            "f " +
            (f.v1 + off) + "// " +
            (f.v2 + off) + "// " +
            (f.v3 + off) + "//"
          );
        });
        off += m.vertices.length;
      });
      return obj.concat([""], verts, [""], faces).join(nl);
    }
  };

  return new Parser();
}());

// ─── Node.js export ──────────────────────────────────────────────
if (typeof module !== 'undefined' && module.exports) {
  module.exports = DATParser;
}
