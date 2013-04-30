(function() {
    var samplers = (function() {
        var str = "";
        var dims = ["x", "y"];
        for (var val = 0; val < 2; val++) {
            var dim = dims[val];
            str += [
            "vec2 make_uv_"+dim+"(vec2 coord, float slice) {",
                "vec2 pos = vec2(mod(slice, mosaic["+val+"].x), floor(slice / mosaic["+val+"].x));",
                "vec2 offset = (pos * (shape["+val+"]+1.)) + 1.;",
                "vec2 imsize = (mosaic["+val+"] * (shape["+val+"]+1.)) + 1.;",
                "return (2.*(offset+coord)+1.) / (2.*imsize);",
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
                "return fract(coord.z) < .5 ? f : c;",
                // "return s*c + (1.-s)*f;",
            "}\n",

            "vec4 debug_"+dim+"(sampler2D data, vec4 fid) {",
                "vec3 coord = (volxfm["+val+"] * fid).xyz;",
                "return vec4(coord / vec3(100., 100., 32.), 1.);",
            "}",


            ].join("\n");
        }
        return str;
    })();

    this.makeShader = function makeShader(sampler, raw, voxline, volume) {
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
        // "attribute vec3 wmnorm;",
        // "attribute float dropout;",

        "varying vec3 vViewPosition;",
        "varying vec3 vNormal;",
        "varying float vCurv;",
        // "varying float vDrop;",
        "varying float vMedial;",
        "varying vec3 vPos[2];",

        "void main() {",

            "vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );",
            "vViewPosition = -mvPosition.xyz;",

            THREE.ShaderChunk[ "map_vertex" ],

            // "vDrop = dropout;",
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
        // "varying float vDrop;",
        "varying float vMedial;",
        "varying vec3 vPos[2];",

        "float rand(vec2 co){",
        "    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);",
        "}",

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
            "vec4 values = vec4(0.);",
            "#endif",
            "",
        ].join("\n");

        
        var sampling = [
    "#ifdef RAWCOLORS",
            "color[0] += "+factor+"*"+sampler+"_x(data[0], fid);",
            "color[1] += "+factor+"*"+sampler+"_x(data[2], fid);",
    "#else",
            "values.x += "+factor+"*"+sampler+"_x(data[0], fid).r;",
            "values.z += "+factor+"*"+sampler+"_x(data[2], fid).r;",
            "values.y += "+factor+"*"+sampler+"_y(data[1], fid).r;",
            "values.w += "+factor+"*"+sampler+"_y(data[3], fid).r;",
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
            fragMid += "vec2 rseed;\n";
            for (var i = 0; i < volume; i++) {
                fragMid += [
                    "rseed = gl_FragCoord.xy + vec2(2.34*"+i.toFixed(3)+", 3.14*"+i.toFixed(3)+");",
                    "fid = vec4(mix(vPos[0], vPos[1], rand(rseed)), 1.);",
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
            "float norm0 = (values.x - vmin[0]) / range;",
            "float norm1 = (values.z - vmin[0]) / range;",
            "float fnorm0 = mix(norm0, norm1, framemix);",

            "range = vmax[1] - vmin[1];",
            "norm0 = (values.y - vmin[1]) / range;",
            "norm1 = (values.w - vmin[1]) / range;",
            "float fnorm1 = mix(norm0, norm1, framemix);",
            "vec2 cuv = vec2(clamp(fnorm0, 0., .999), clamp(fnorm1, 0., .999) );",

            "vec4 vColor = texture2D(colormap, cuv);",
            "bvec4 valid = notEqual(lessThanEqual(values, vec4(0.)), lessThan(vec4(0.), values));",
            "vColor = all(valid) ? vColor : vec4(0.);",
        "#endif",
            "vColor *= dataAlpha;",

    "#ifdef VOXLINE",
        "#ifdef CORTSHEET",
            "fid = vec4(mix(vPos[0], vPos[1], 0.5),1.);",
        "#else",
            "fid = vec4(vPos[0], 1.);",
        "#endif",
            "vec3 coord = (volxfm[0]*fid).xyz;",
            "vec3 edge = abs(fract(coord) - vec3(0.5));",
            "vColor = mix(vec4(voxlineColor, 1.), vColor, edgeFactor(edge));",
    "#endif",

            //Cross hatch / dropout layer
            // "float hw = gl_FrontFacing ? hatchAlpha*vDrop : 1.;",
            // "vec4 hColor = hw * vec4(hatchColor, 1.) * texture2D(hatch, vUv*hatchrep);",

            //roi layer
            "vec4 rColor = texture2D(map, vUv);",

            "if (vMedial < .999) {",
                "gl_FragColor = cColor;",
                "gl_FragColor = vColor + (1.-vColor.a)*gl_FragColor;",
                // "gl_FragColor = hColor + (1.-hColor.a)*gl_FragColor;",
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

})();