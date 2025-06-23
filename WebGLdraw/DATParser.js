var DATParser = (function () {
  "use strict";

  // Node.js support for reading .dat files from disk
  var fs   = (typeof require === 'function') ? require('fs') : null;
  var path = (typeof require === 'function') ? require('path') : null;

  // Cache for browser fallback
  var dataCache = {};

  // Build an index of all .dat files under the 'parts' directory for fast lookup
  var datIndex = {};
  if (fs) {
    var partsRoot = path.join(__dirname, 'parts');
    function indexDir(dir) {
      fs.readdirSync(dir).forEach(function(name) {
        var full = path.join(dir, name);
        var stat = fs.statSync(full);
        if (stat.isDirectory()) {
          indexDir(full);
        } else if (name.toLowerCase().endsWith('.dat')) {
          datIndex[name.toLowerCase()] = full;
        }
      });
    }
    if (fs.existsSync(partsRoot) && fs.statSync(partsRoot).isDirectory()) {
      indexDir(partsRoot);
    }
  }

  /**
   * Get the data for a .dat file, using our prebuilt index
   */
  var getDATFile = function(filename) {
    // 1) Strip surrounding quotes
    var fn = filename.replace(/^['\"]+|['\"]+$/g, '');
    // 2) Normalize backslashes to forward slashes
    var norm = fn.replace(/\\/g, '/');
    // 3) Base filename lowercased
    var base = path.basename(norm).toLowerCase();

    if (fs) {
      // 4) Indexed lookup
      if (datIndex[base]) {
        return fs.readFileSync(datIndex[base], 'utf8');
      }
      // 5) Flattened slash variant: 's/50956s01.dat' -> 's50956s01.dat'
      if (norm.includes('/')) {
        var flat = norm.split('/').join('').toLowerCase();
        if (datIndex[flat]) {
          return fs.readFileSync(datIndex[flat], 'utf8');
        }
      }
      // 6) Direct file path fallback
      if (fs.existsSync(norm) && fs.statSync(norm).isFile()) {
        return fs.readFileSync(norm, 'utf8');
      }
      throw new Error('DATParser: file not found: ' + filename);
    }

    // Browser fallback via synchronous XHR
    if (!dataCache[filename]) {
      var xhr = new XMLHttpRequest();
      xhr.open('GET', filename, false);
      xhr.send(null);
      if (xhr.status === 200 && xhr.responseText.trim() !== '') {
        dataCache[filename] = xhr.responseText;
      } else {
        throw new Error('DATParser (XHR) could not load ' + filename);
      }
    }
    return dataCache[filename];
  };

  /**
   * Cast a string to float
   */
  var castFloat = function(v) { return parseFloat(v); };

  // Unique mesh ID generator
  var uid = 1;

  /**
   * Build a new mesh object for collecting vertices and faces.
   */
  var buildMesh = function() {
    return {
      meshId: uid++,
      vertices: [],
      faces: [],
      addVertex: function(x, y, z, c) {
        this.vertices.push({x: x, y: y, z: z, color: c});
      },
      addFace: function(m, c, clockwise, inverted) {
        var v1 = this.vertices.length;
        this.addVertex(m[0], m[1], m[2], c);
        var v2 = this.vertices.length;
        this.addVertex(m[3], m[4], m[5], c);
        var v3 = this.vertices.length;
        this.addVertex(m[6], m[7], m[8], c);
        var normal = (clockwise && inverted) || (!clockwise && !inverted);
        this.faces.push({v1: v1, v2: (normal ? v2 : v3), v3: (normal ? v3 : v2)});
      },
      addQuad: function(m, c, clockwise, inverted) {
        var v1 = this.vertices.length;
        this.addVertex(m[0], m[1], m[2], c);
        var v2 = this.vertices.length;
        this.addVertex(m[3], m[4], m[5], c);
        var v3 = this.vertices.length;
        this.addVertex(m[6], m[7], m[8], c);
        var v4 = this.vertices.length;
        this.addVertex(m[9], m[10], m[11], c);
        var normal = (clockwise && inverted) || (!clockwise && !inverted);
        this.faces.push({v1: v1, v2: (normal ? v2 : v3), v3: (normal ? v3 : v2)});
        this.faces.push({v1: v3, v2: (normal ? v4 : v1), v3: (normal ? v1 : v4)});
      },
      apply: function(offset, mat, color) {
        this.vertices.forEach(function(v) {
          var x = mat[0] * v.x + mat[1] * v.y + mat[2] * v.z + offset[0];
          var y = mat[3] * v.x + mat[4] * v.y + mat[5] * v.z + offset[1];
          var z = mat[6] * v.x + mat[7] * v.y + mat[8] * v.z + offset[2];
          v.x = x; v.y = y; v.z = z; v.color = color;
        });
      }
    };
  };

  /**
   * Parser for .dat instructions
   */
  var Parser = function(inverted, consolePrefix) {
    this.invertNext = false;
    this.clockwise = false;
    this.invert = inverted;
    this.consolePrefix = consolePrefix || '';
  };

  Parser.prototype = {
    log: (console && console.log) ? function(s) { console.log(this.consolePrefix + s); } : function() {},
    handleBFC: function(ops) {
      ops.forEach(function(op) {
        if (op === 'CW') this.clockwise = true;
        if (op === 'CCW') this.clockwise = false;
        if (op === 'INVERTNEXT') this.invertNext = true;
      }.bind(this));
    },
    handleComment: function(ops) {
      if (ops[0] === 'BFC') { ops.shift(); this.handleBFC(ops); }
    },
    handleDependency: function(ops, meshes) {
      var dep = ops.pop();
      ops = ops.map(castFloat);
      var det = (function(m) {
        return m[0] * (m[4] * m[8] - m[7] * m[5])
             - m[3] * (m[1] * m[8] - m[7] * m[2])
             + m[6] * (m[1] * m[5] - m[4] * m[2]);
      })(ops.slice(4, 13));
      var inv = this.invert;
      if (this.invertNext) inv = !inv;
      if (det < 0) inv = !inv;
      this.invertNext = false;
      var childParser = new Parser(inv, this.consolePrefix);
      var childMeshes = childParser.parseFile(dep);
      childMeshes.forEach(function(mesh) {
        mesh.apply(ops.slice(1, 4), ops.slice(4, 13), ops[0]);
        meshes.push(mesh);
      });
    },
    parseTriangle: function(args, mesh) {
      var a = args.map(castFloat);
      mesh.addFace(a.slice(1), a[0], this.clockwise, this.invert);
    },
    parseQuad: function(args, mesh) {
      var a = args.map(castFloat);
      mesh.addQuad(a.slice(1), a[0], this.clockwise, this.invert);
    },
    parseInstruction: function(line, meshes) {
      var cur = meshes[meshes.length - 1];
      if (!line.trim() && cur.vertices.length) { meshes.push(buildMesh()); return; }
      var ops = line.trim().split(/\s+/);
      var op = ops.shift();
      if (op === '0') this.handleComment(ops);
      else if (op === '1') this.handleDependency(ops, meshes);
      else if (op === '3') this.parseTriangle(ops, cur);
      else if (op === '4') this.parseQuad(ops, cur);
    },
    parse: function(filename, data) {
      this.log(' * parsing ' + filename);
      var lines = data.split(/\r?\n/);
      var meshes = [buildMesh()];
      lines.forEach(function(ln) { this.parseInstruction(ln, meshes); }.bind(this));
      if (!meshes[meshes.length - 1].vertices.length) meshes.pop();
      return meshes;
    },
    parseFile: function(filename) {
      var data = getDATFile(filename);
      return this.parse(filename, data);
    },
    toOBJ: function(meshData, scale) {
      scale = scale || 0.05;
      var nl = '\n';
      var obj = ['# LDraw .dat to OBJ'];
      var verts = ['# vertices:'];
      var faces = ['# faces:'];
      var offset = 1;
      var prec = 6;
      var nf = (function(p) { var m = Math.pow(10, p); return function(v) { return ((v * m) | 0) / m; }; }(prec));
      meshData.forEach(function(mesh) {
        if (!mesh.vertices.length) return;
        verts.push('', '# mesh ' + mesh.meshId);
        mesh.vertices.forEach(function(v) {
          verts.push('v ' + nf(v.x * scale) + ' ' + nf(v.y * scale) + ' ' + nf(v.z * scale));
        });
        faces.push('g mesh' + mesh.meshId);
        faces.push('usemtl color' + mesh.vertices[0].color);
        faces.push('s 1', '');
        mesh.faces.forEach(function(f) {
          faces.push('f ' + (f.v1 + offset) + ' ' + (f.v2 + offset) + ' ' + (f.v3 + offset));
        });
        offset += mesh.vertices.length;
        faces.push('');
      });
      return obj.concat([''], verts, [''], faces).join(nl);
    }
  };

  return new Parser();
}());

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = DATParser;
}
