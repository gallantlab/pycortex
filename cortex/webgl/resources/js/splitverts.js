function splitverts(geom, left_off) {
    var o, ol, i, il, j, jl, k, n = 0, stride, mpts, faceidx, attr, idx, pos;
    var npolys = geom.attributes.index.array.length;
    var newgeom = new THREE.BufferGeometry();
    for (var name in geom.attributes) {
        if (name != "index") {
            attr = geom.attributes[name];
            newgeom.attributes[name] = {itemSize:attr.itemSize, stride:attr.itemSize};
            newgeom.attributes[name].array = new Float32Array(npolys*attr.itemSize);
        }
    }
    newgeom.attributes.index = {itemSize:1, array:new Uint16Array(npolys), stride:1};
    newgeom.attributes.face = {itemSize:1, array:new Float32Array(npolys), stride:1};
    newgeom.attributes.bary = {itemSize:3, array:new Float32Array(npolys*3), stride:3};
    newgeom.attributes.vert = {itemSize:3, array:new Float32Array(npolys*3), stride:3};
    newgeom.morphTargets = [];
    newgeom.morphNormals = [];
    for (i = 0, il = geom.morphTargets.length; i < il; i++) {
        newgeom.morphTargets.push({itemSize:3, array:new Float32Array(npolys*3), stride:3});
        newgeom.morphNormals.push(new Float32Array(npolys*3));
    }
    newgeom.offsets = []
    for (i = 0, il = npolys; i < il; i += 65535) {
        newgeom.offsets.push({start:i, index:i, count:Math.min(65535, il - i)});
    }

    for (o = 0, ol = geom.offsets.length; o < ol; o++) {
        var start = geom.offsets[o].start;
        var count = geom.offsets[o].count;
        var index = geom.offsets[o].index;

        //For each face
        for (j = start, jl = start+count; j < jl; j+=3) {
            //For each vertex
            for (k = 0; k < 3; k++) {
                idx = index + geom.attributes.index.array[j+k];

                for (var name in geom.attributes) {
                    if (name != "index") {
                        attr = geom.attributes[name];
                        for (i = 0, il = attr.itemSize; i < il; i++)
                            newgeom.attributes[name].array[n*il+i] = attr.array[idx*attr.stride+i];
                    }
                }

                for (i = 0, il = geom.morphTargets.length; i < il; i++) {
                    stride = geom.morphTargets[i].stride;
                    newgeom.morphTargets[i].array[n*3+0] = geom.morphTargets[i].array[idx*stride+0];
                    newgeom.morphTargets[i].array[n*3+1] = geom.morphTargets[i].array[idx*stride+1];
                    newgeom.morphTargets[i].array[n*3+2] = geom.morphTargets[i].array[idx*stride+2];
                    newgeom.morphNormals[i][n*3+0] = geom.morphNormals[i][idx*3+0];
                    newgeom.morphNormals[i][n*3+1] = geom.morphNormals[i][idx*3+1];
                    newgeom.morphNormals[i][n*3+2] = geom.morphNormals[i][idx*3+2];
                }

                newgeom.attributes.vert.array[n*3+0] = left_off + index + geom.attributes.index.array[j+0];
                newgeom.attributes.vert.array[n*3+1] = left_off + index + geom.attributes.index.array[j+1];
                newgeom.attributes.vert.array[n*3+2] = left_off + index + geom.attributes.index.array[j+2];

                newgeom.attributes.face.array[n] = j / 3;
                newgeom.attributes.bary.array[n*3+k] = 1;
                newgeom.attributes.index.array[n] = n % 65535;
                n++;
            }
        }
    }
    newgeom.computeBoundingBox();
    return newgeom;
}

function makeFlat(uv, flatlims, flatoff, right) {
    var fmin = flatlims[0], fmax = flatlims[1];
    var flat = new Float32Array(uv.length / 2 * 3);
    var norms = new Float32Array(uv.length / 2 * 3);
    for (var i = 0, il = uv.length / 2; i < il; i++) {
        if (right) {
            flat[i*3+1] = flatscale*uv[i*2] + flatoff[1];
            norms[i*3] = 1;
        } else {
            flat[i*3+1] = flatscale*-uv[i*2] + flatoff[1];
            norms[i*3] = -1;
        }
        flat[i*3+2] = flatscale*uv[i*2+1];
        uv[i*2]   = (uv[i*2]   + fmin[0]) / fmax[0];
        uv[i*2+1] = (uv[i*2+1] + fmin[1]) / fmax[1];
    }

    return {pos:flat, norms:norms};
}

function makeHatch(size, linewidth, spacing) {
    //Creates a cross-hatching pattern in canvas for the dropout shading
    if (spacing === undefined)
        spacing = 32;
    if (size === undefined)
        size = 128;
    var canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    var ctx = canvas.getContext("2d");
    ctx.lineWidth = linewidth ? linewidth: 8;
    ctx.strokeStyle = "white";
    for (var i = 0, il = size*2; i < il; i+=spacing) {
        ctx.beginPath();
        ctx.moveTo(i-size, 0);
        ctx.lineTo(i, size);
        ctx.stroke();
        ctx.moveTo(i, 0)
        ctx.lineTo(i-size, size);
        ctx.stroke();
    }

    var tex = new THREE.Texture(canvas);
    tex.needsUpdate = true;
    tex.premultiplyAlpha = true;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearMipMapLinearFilter;
    return tex;
};

function makeShader(sampler, raw, voxline, volume) {
    //Creates shader code with all the parameters
    //sampler: which sampler to use, IE nearest or trilinear
    //raw: whether the dataset is raw or not
    //voxline: whether to show the voxel lines
    //volume: the number of volume samples to take

    if (volume === undefined)
        volume = 0;

    var header = "";
    if (voxline)
        header += "#define VOXLINE\n";
    if (raw)
        header += "#define RAWCOLORS\n";
    if (volume > 0)
        header += "#define CORTSHEET\n";

    var vertShade =  [
    THREE.ShaderChunk[ "map_pars_vertex" ], 
    THREE.ShaderChunk[ "lights_phong_pars_vertex" ],
    THREE.ShaderChunk[ "morphtarget_pars_vertex" ],

    "uniform float thickmix;",

    "attribute vec4 auxdat;",
    "attribute vec3 wm;",

    "varying vec3 vViewPosition;",
    "varying vec3 vNormal;",
    "varying float vCurv;",
    "varying float vDrop;",
    "varying float vMedial;",
    "varying vec3 vPos[2];",

    "void main() {",

        "vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );",
        "vViewPosition = -mvPosition.xyz;",

        THREE.ShaderChunk[ "map_vertex" ],

        "vDrop = auxdat.x;",
        "vCurv = auxdat.y;",
        "vMedial = auxdat.z;",

        "vPos[0] = position;",
        "#ifdef CORTSHEET",
        "vPos[1] = wm;",
        "vec3 npos = mix(position, wm, thickmix);",
        "#else",
        "vec3 npos = position;",
        "#endif",

        THREE.ShaderChunk[ "morphnormal_vertex" ],
        "vNormal = transformedNormal;",

        "vec3 morphed = vec3( 0.0 );",
        "morphed += ( morphTarget0 - npos ) * morphTargetInfluences[ 0 ];",
        "morphed += ( morphTarget1 - npos ) * morphTargetInfluences[ 1 ];",
        "morphed += ( morphTarget2 - npos ) * morphTargetInfluences[ 2 ];",
        "morphed += ( morphTarget3 - npos ) * morphTargetInfluences[ 3 ];",
        "morphed += npos;",
        "gl_Position = projectionMatrix * modelViewMatrix * vec4( morphed, 1.0 );",

    "}"
    ].join("\n");

    var samplers = "";
    var dims = ["x", "y"];
    for (var val = 0; val < 2; val++) {
        var dim = dims[val];
        samplers += [
        "vec2 make_uv_"+dim+"(vec2 coord, float slice) {",
            "vec2 pos = vec2(mod(slice, mosaic["+val+"].x), floor(slice / mosaic["+val+"].x));",
            "return (2.*(pos*shape["+val+"]+coord)+1.) / (2.*mosaic["+val+"]*shape["+val+"]);",
        "}",

        "vec4 trilinear_"+dim+"(sampler2D data, vec4 fid) {",
            "vec3 coord = (volxfm["+val+"] * fid).xyz;",
            "vec4 tex1 = texture2D(data, make_uv_"+dim+"(coord.xy, floor(coord.z)));",
            "vec4 tex2 = texture2D(data, make_uv_"+dim+"(coord.xy, ceil(coord.z)));",
            'return mix(tex1, tex2, fract(coord.z));',
        "}",

        "vec4 nearest_"+dim+"(sampler2D data, vec4 fid) {",
            "vec3 coord = (volxfm["+val+"]*fid).xyz;",
            "float s = step(0.5, fract(coord.z));",
            "vec4 c = texture2D(data, make_uv_"+dim+"(coord.xy, ceil(coord.z)));",
            "vec4 f = texture2D(data, make_uv_"+dim+"(coord.xy, floor(coord.z)));",
            "return s*c + (1.-s)*f;",
        "}\n",

        "vec4 debug_"+dim+"(sampler2D data, vec4 fid) {",
            "vec3 coord = (volxfm["+val+"] * fid).xyz;",
            "return vec4(coord / vec3(100., 100., 32.), 1.);",
        "}",


        ].join("\n");
    }

    var factor = volume > 1 ? (1/volume).toFixed(6) : "1.";

    var fragHead = [
    "#extension GL_OES_standard_derivatives: enable",
    "#extension GL_OES_texture_float: enable",
    THREE.ShaderChunk[ "map_pars_fragment" ],
    THREE.ShaderChunk[ "lights_phong_pars_fragment" ],

    "uniform int hide_mwall;",

    "uniform vec3 diffuse;",
    "uniform vec3 ambient;",
    "uniform vec3 emissive;",
    "uniform vec3 specular;",
    "uniform float shininess;",

    "uniform sampler2D colormap;",
    "uniform float vmin[2];",
    "uniform float vmax[2];",
    "uniform float framemix;",

    "uniform float curvAlpha;",
    "uniform float curvScale;",
    "uniform float curvLim;",
    "uniform float dataAlpha;",
    "uniform float hatchAlpha;",
    "uniform vec3 hatchColor;",
    "uniform vec3 voxlineColor;",
    "uniform float voxlineWidth;",

    "uniform sampler2D hatch;",
    "uniform vec2 hatchrep;",

    "uniform mat4 volxfm[2];",
    "uniform vec2 mosaic[2];",
    "uniform vec2 shape[2];",
    "uniform sampler2D data[4];",
    "uniform float thickmix;",

    "varying float vCurv;",
    "varying float vDrop;",
    "varying float vMedial;",
    "varying vec3 vPos[2];",

    // from http://codeflow.org/entries/2012/aug/02/easy-wireframe-display-with-barycentric-coordinates/
    "float edgeFactor(vec3 edge) {",
        "vec3 d = fwidth(edge)+voxlineWidth;",
        "vec3 a3 = smoothstep(vec3(0.), d, edge);",
        "return min(min(a3.x, a3.y), a3.z);",
    "}",

    samplers,

    "void main() {",
        //Curvature Underlay
        "float curv = clamp(vCurv / curvScale  + .5, curvLim, 1.-curvLim);",
        "vec4 cColor = vec4(vec3(curv) * curvAlpha, curvAlpha);",

        "vec4 fid;",
        "#ifdef RAWCOLORS",
        "vec4 color[2]; color[0] = vec4(0.), color[1] = vec4(0.);",
        "#else",
        "float val[4]; val[0] = 0.; val[1] = 0.; val[2] = 0.; val[3] = 0.;",
        "#endif",
        "",
    ].join("\n");

    
    var sampling = [
"#ifdef RAWCOLORS",
        "color[0] += "+factor+"*"+sampler+"_x(data[0], fid);",
        "color[1] += "+factor+"*"+sampler+"_x(data[2], fid);",
"#else",
        "val[0] += "+factor+"*"+sampler+"_x(data[0], fid).r;",
        "val[2] += "+factor+"*"+sampler+"_x(data[2], fid).r;",
        "val[1] += "+factor+"*"+sampler+"_y(data[1], fid).r;",
        "val[3] += "+factor+"*"+sampler+"_y(data[3], fid).r;",
"#endif",
    ].join("\n");

    var fragMid = "";
    if (volume == 0) {
        fragMid += [
            "fid = vec4(vPos[0], 1.);",
            sampling,
            ""
        ].join("\n");
    } else if (volume == 1) {
        fragMid += [
            "fid = vec4(mix(vPos[0], vPos[1], thickmix), 1.);",
            sampling,
            "",
        ].join("\n");
    } else {
        for (var i = 0; i < volume; i++) {
            fragMid += [
                "fid = vec4(mix(vPos[0], vPos[1], "+i+". / "+(volume-1)+".), 1.);",
                sampling,
                "", 
            ].join("\n");
        }
    }

    var fragTail = [
    "#ifdef RAWCOLORS",
        "vec4 vColor = mix(color[0], color[1], framemix);",
    "#else",
        "float range = vmax[0] - vmin[0];",
        "float norm0 = (val[0] - vmin[0]) / range;",
        "float norm1 = (val[2] - vmin[0]) / range;",
        "float fnorm0 = mix(norm0, norm1, framemix);",

        "range = vmax[1] - vmin[1];",
        "norm0 = (val[1] - vmin[1]) / range;",
        "norm1 = (val[3] - vmin[1]) / range;",
        "float fnorm1 = mix(norm0, norm1, framemix);",
        "vec2 cuv = vec2(clamp(fnorm0, 0., .999), clamp(fnorm1, 0., .999) );",

        "vec4 vColor = texture2D(colormap, cuv);",
    "#endif",

    "#ifdef VOXLINE",
        "vec3 coord = (volxfm[0]*fid).xyz;",
        "vec3 edge = abs(mod(coord, 1.) - vec3(0.5));",
        "vColor = mix(vec4(voxlineColor, 1.), vColor, edgeFactor(edge));",
    "#endif",

        //Cross hatch / dropout layer
        "float hw = gl_FrontFacing ? hatchAlpha*vDrop : 1.;",
        "vec4 hcolor = hw * vec4(hatchColor, 1.) * texture2D(hatch, vUv*hatchrep);",

        //roi layer
        "vec4 rColor = texture2D(map, vUv);",

        "if (vMedial < .999) {",
            "gl_FragColor = mix(cColor, vColor, dataAlpha);",
            "gl_FragColor = hcolor + (1.-hcolor.a)*gl_FragColor;",
            "gl_FragColor = rColor + (1.-rColor.a)*gl_FragColor;",
        "} else if (hide_mwall == 1) {",
            "discard;",
        "} else {",
            "gl_FragColor = cColor;",
        "}",

        THREE.ShaderChunk[ "lights_phong_fragment" ],
    "}"
    ].join("\n");


    return {vertex:header+vertShade, fragment:header+fragHead+fragMid+fragTail};
}