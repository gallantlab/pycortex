var Shaderlib = (function() {

    var utils = {
        rand: [
            "float rand(vec2 co){",
            "    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);",
            "}",
        ].join("\n"),
        edge: [
            // from http://codeflow.org/entries/2012/aug/02/easy-wireframe-display-with-barycentric-coordinates/
            "float edgeFactor(vec3 edge) {",
                "vec3 d = fwidth(edge)+voxlineWidth;",
                "vec3 a3 = smoothstep(vec3(0.), d, edge);",
                "return min(min(a3.x, a3.y), a3.z);",
            "}",
        ].join("\n"),

        colormap: [
            "vec2 vnorm(vec4 values) {",
                "float range = vmax[0] - vmin[0];",
                "float norm0 = (values.x - vmin[0]) / range;",
                "float norm1 = (values.y - vmin[0]) / range;",
                "float fnorm0 = mix(norm0, norm1, framemix);",
            "#ifdef TWOD",
                "range = vmax[1] - vmin[1];",
                "norm0 = (values.z - vmin[1]) / range;",
                "norm1 = (values.w - vmin[1]) / range;",
                "float fnorm1 = mix(norm0, norm1, framemix);",
                "vec2 cuv = vec2(clamp(fnorm0, 0., 1.), clamp(fnorm1, 0., 1.) );",
            "#else",
                "vec2 cuv = vec2(clamp(fnorm0, 0., 1.), 0.);",
            "#endif",
                "return cuv;",
            "}",
            "vec4 colorlut(vec4 values) {",
                "vec2 cuv = vnorm(values);",
                "vec4 vColor = texture2D(colormap, cuv);",
                "bvec4 valid = notEqual(lessThanEqual(values, vec4(0.)), lessThan(vec4(0.), values));",
                "return all(valid) ? vColor : vec4(0.);",
            "}",
        ].join("\n"),

        pack: [
            //Stores float into 32 bits, 4 bytes
            "vec4 encode_float32( const in float depth ) {",
                "const vec4 bit_shift = vec4( 1.0, 255.0, 65025.0, 160581375.0 );",
                "const vec4 bit_mask  = vec4( 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0);",
                "vec4 res = fract( depth * bit_shift );",
                "res -= res.yzww * bit_mask;",
                "return res;",
            "}",
            //stores float into 16 bits, 2 bytes
            "vec2 encode_float16( const in float depth ) {",
                "const vec2 bit_shift = vec2( 1.0, 255.0);",
                "const vec2 bit_mask  = vec2( 1.0 / 255.0, 0.0);",
                "vec2 res = fract( depth * bit_shift );",
                "res -= res.yy * bit_mask;",
                "return res;",
            "}",
            //stores float into 8 bits, 4bits/4bits in each
            "vec2 encode_float8_(const in float value) {",
                "const vec2 bit_shift = vec2(1., 15.);",
                "const vec2 bit_mask = vec2(15./255., 0.);",
                "vec2 res = fract(value * bit_shift);",
                "res -= res.yy * bit_mask;",
                "return res * bit_mask.xx;",
            "}",
            "float decode_float_vec4(vec4 rgba) {",
                "const vec4 shift = vec4(1.0, 1. / 255., 1. / 65025., 1./160581375.0);",
                "return dot(rgba, shift);",
            "}",
            "float decode_float_vec2(vec2 rg) {",
                "const vec2 shift = vec2(1.0, 1. / 255.);",
                "return dot(rg, shift);",
            "}",

            "vec4 pack_float( const in float depth ) {",
                "const vec4 bit_shift = vec4( 256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0 );",
                "const vec4 bit_mask  = vec4( 0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0 );",
                "vec4 res = fract( depth * bit_shift );",
                "res -= res.xxyz * bit_mask;",
                "return res;",
            "}",
        ].join("\n"),

        samplers: [
            "vec2 make_uv_x(vec2 coord, float slice) {",
                "vec2 pos = vec2(mod(slice+.5, mosaic[0].x), (slice+.5) / mosaic[0].x);",
                "vec2 offset = (floor(pos) * (dshape[0]+1.)) + 1.;",
                "vec2 imsize = (mosaic[0] * (dshape[0]+1.)) + 1.;",
                "return (2.*(offset+coord)+1.) / (2.*imsize);",
            "}",

            "vec4 trilinear_x(sampler2D data, vec3 coord) {",
                "vec4 tex1 = texture2D(data, make_uv_x(coord.xy, floor(coord.z)));",
                "vec4 tex2 = texture2D(data, make_uv_x(coord.xy, ceil(coord.z)));",
                'return mix(tex1, tex2, fract(coord.z));',
            "}",

            "vec4 nearest_x(sampler2D data, vec3 coord) {",
                "return texture2D(data, make_uv_x(coord.xy, floor(coord.z+.5)));",
            "}",

            "vec4 debug_x(sampler2D data, vec3 coord) {",
                "return vec4(coord / vec3(136., 136., 38.), 1.);",
            "}",

            "vec2 make_uv_y(vec2 coord, float slice) {",
                "vec2 pos = vec2(mod(slice+.5, mosaic[1].x), (slice+.5) / mosaic[1].x);",
                "vec2 offset = (floor(pos) * (dshape[1]+1.)) + 1.;",
                "vec2 imsize = (mosaic[1] * (dshape[1]+1.)) + 1.;",
                "return (2.*(offset+coord)+1.) / (2.*imsize);",
            "}",

            "vec4 trilinear_y(sampler2D data, vec3 coord) {",
                "vec4 tex1 = texture2D(data, make_uv_y(coord.xy, floor(coord.z)));",
                "vec4 tex2 = texture2D(data, make_uv_y(coord.xy, ceil(coord.z)));",
                'return mix(tex1, tex2, fract(coord.z));',
            "}",

            "vec4 nearest_y(sampler2D data, vec3 coord) {",
                "return texture2D(data, make_uv_y(coord.xy, floor(coord.z+.5)));",
            "}",

            "vec4 debug_y(sampler2D data, vec3 coord) {",
                "return vec4(coord / vec3(100., 100., 32.), 1.);",
            "}",
        ].join("\n"),

        standard_frag_vars: [
            "uniform vec3 diffuse;",
            "uniform vec3 ambient;",
            "uniform vec3 emissive;",
            "uniform vec3 specular;",
            "uniform float shininess;",
            "uniform float specularStrength;",

            "uniform float dataAlpha;",
        ].join("\n"),

        mixer: function(morphs) {
            var glsl = "uniform float surfmix;\n";
            for (var i = 0; i < morphs-1; i++) {
                glsl += "attribute vec4 mixSurfs"+i+";\n";
                glsl += "attribute vec3 mixNorms"+i+";\n";
            }
            glsl += [
            "void mixfunc(vec3 basepos, vec3 basenorm, out vec3 pos, out vec3 norm) {",
                "float smix = surfmix * "+(morphs-1)+".;",
                "float factor = clamp(1. - smix, 0., 1.);",
                "pos = factor * basepos;",
                "norm = factor * basenorm;",
                "",
            ].join("\n");
            for (var i = 0; i < morphs-1; i++) {
                glsl += "factor = clamp( 1. - abs(smix - "+(i+1)+".) , 0., 1.);\n";
                glsl += "pos  += factor * mixSurfs"+i+".xyz;\n";
                glsl += "norm += factor * mixNorms"+i+";\n";
            }
            glsl += [ "",
            "}",
            "void mixfunc_pos(vec3 basepos, out vec3 pos) {",
                "float smix = surfmix * "+(morphs-1)+".;",
                "float factor = clamp(1. - smix, 0., 1.);",
                "pos = factor * basepos;",
            ].join("\n");
            for (var i = 0; i < morphs-1; i++) {
                glsl += "factor = clamp( 1. - abs(smix - "+(i+1)+".) , 0., 1.);\n";
                glsl += "pos  += factor * mixSurfs"+i+".xyz;\n";
            }
            glsl += [ "",
            "}",
            ].join("\n");
            return glsl;
        },

        // thickmixer: header code that loads the uniforms and attributes needed to
        // do equivolume sampling, for vertex shaders
        thickmixer: [
            "uniform float thickmix;",
            "uniform int equivolume;",
            "attribute float wmarea;",
            "attribute float pialarea;",
        ].join("\n"),

        // thickmixer_main: translates a desired volume fraction into linear mixing
        // parameter.
        thickmixer_main: [
            "#ifdef EQUIVOLUME",
                "float use_thickmix = 1. - (1. / (pialarea - wmarea) * (-1. * wmarea + sqrt((1. - thickmix) * pialarea * pialarea + thickmix * wmarea * wmarea)));",
            "#else",
                "float use_thickmix = thickmix;",
            "#endif",
        ].join("\n"),
    }

    var module = function() {

    };
    module.prototype = {
        constructor: module,
        main: function(opts) {
            //Creates shader code with all the parameters
            //sampler: which sampler to use, IE nearest or trilinear
            //raw: whether the dataset is raw or not
            //voxline: whether to show the voxel lines
            
            var sampler = opts.sampler;
            var header = "";
            if (opts.voxline)
                header += "#define VOXLINE\n";
            if (opts.raw)
                header += "#define RGBCOLORS\n";
            if (opts.twod)
                header += "#define TWOD\n";
            if (!opts.viewspace)
                header += "#define SAMPLE_WORLD\n";
            if (opts.lights !== undefined && !opts.lights)
                header += "#define NOLIGHTS\n";

            var vertShade =  [
        "#ifndef NOLIGHTS",
            THREE.ShaderChunk[ "lights_phong_pars_vertex" ],
        "#endif",

            "uniform mat4 volxfm[2];",

            "attribute vec4 auxdat;",

            "varying vec3 vViewPosition;",
            "varying vec3 vNormal;",

            "varying vec3 vPos_x;",
        "#ifdef TWOD",
            "varying vec3 vPos_y;",
        "#endif",

            "void main() {",

                "vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );",
                "vViewPosition = -mvPosition.xyz;",

                //Find voxel positions with both transforms (2D colormap x and y datasets)
        "#ifdef SAMPLE_WORLD",
                "vec4 sample = vec4(position, 1.);",
        "#else",
                "vec4 sample = modelMatrix * vec4(position, 1.);",
        "#endif", 
                "vPos_x = (volxfm[0]*sample).xyz;",
        "#ifdef TWOD",
                "vPos_y = (volxfm[1]*sample).xyz;",
        "#endif",

                "vNormal = normalMatrix * normal;",
                "gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0);",

            "}"
            ].join("\n");

            var fragShade = [
            "#extension GL_OES_standard_derivatives: enable",

            "uniform vec3 voxlineColor;",
            "uniform float voxlineWidth;",

            "uniform sampler2D colormap;",
            "uniform float vmin[2];",
            "uniform float vmax[2];",
            "uniform float framemix;",
            "uniform vec2 mosaic[2];",
            "uniform vec2 dshape[2];",
            "uniform sampler2D data[4];",

            "varying vec3 vPos_x;",
            "varying vec3 vPos_y;",
        "#ifndef NOLIGHTS",
            THREE.ShaderChunk[ "lights_phong_pars_fragment" ],
        "#endif",
            
            utils.standard_frag_vars,
            utils.rand,
            utils.edge,
            utils.colormap,
            utils.samplers,

            "void main() {",
            "#ifdef RGBCOLORS",
                "vec4 color[2]; color[0] = vec4(0.), color[1] = vec4(0.);",
            "#else",
                "vec4 values = vec4(0.);",
            "#endif",
        
        "#ifdef RGBCOLORS",
                "color[0] += "+sampler+"_x(data[0], vPos_x);",
                "color[1] += "+sampler+"_x(data[1], vPos_x);",
        "#else",
                "values.x += "+sampler+"_x(data[0], vPos_x).r;",
                "values.y += "+sampler+"_x(data[1], vPos_x).r;",
            "#ifdef TWOD",
                "values.z += "+sampler+"_y(data[2], vPos_y).r;",
                "values.w += "+sampler+"_y(data[3], vPos_y).r;",
            "#endif",
        "#endif",
            "#ifdef RGBCOLORS",
                "vec4 vColor = mix(color[0], color[1], framemix);",
            "#else",
                "vec4 vColor = colorlut(values);",
            "#endif",
                "vColor *= dataAlpha;",

        "#ifdef VOXLINE",
                "vec3 coord = vPos_x[0];",
                "vec3 edge = abs(fract(coord) - vec3(0.5));",
                "vColor = mix(vec4(voxlineColor, 1.), vColor, edgeFactor(edge*1.001));",
        "#endif",

                "if (vColor.a < .001) discard;",
                "gl_FragColor = vColor;",
        "#ifndef NOLIGHTS",
                THREE.ShaderChunk[ "lights_phong_fragment" ],
        "#endif",
            "}"
            ].join("\n");

            return {vertex:header+vertShade, fragment:header+fragShade, attrs:{}};
        },

        surface_pixel: function(opts) {
            var header = "";
            if (opts.voxline)
                header += "#define VOXLINE\n";
            if (opts.rgb)
                header += "#define RGBCOLORS\n";
            if (opts.twod)
                header += "#define TWOD\n";

            var sampler = opts.sampler || "nearest";
            var morphs = opts.morphs;
            var volume = opts.volume || 0;
            var layers = opts.layers || 1;
            var dither = opts.dither || false;
            if (volume > 0)
                header += "#define CORTSHEET\n";
            if (opts.rois)
                header += "#define ROI_RENDER\n";
            if (opts.extratex)
                header += "#define EXTRATEX\n"
            if (opts.halo) {
                if (opts.twod)
                    throw "Cannot use 2D colormaps with volume integration"
                header += "#define HALO_RENDER\n";
            }
            if (opts.hasflat)
                header += "#define HASFLAT\n"
            if (opts.equivolume)
                header += "#define EQUIVOLUME\n"

            var vertShade =  [
            THREE.ShaderChunk[ "lights_phong_pars_vertex" ],
            "uniform mat4 volxfm[2];",
            // "uniform float thickmix;",
            utils.thickmixer,
            "uniform int bumpyflat;",
            "float f_bumpyflat = float(bumpyflat);",

            "attribute vec4 wm;",
            "attribute vec3 wmnorm;",
            "attribute vec4 auxdat;",

            "#ifdef HASFLAT",
                "attribute vec3 flatBumpNorms;",
                "attribute float flatheight;",
            "#endif",
            // "attribute float dropout;",
            
            "varying vec3 vViewPosition;",
            "varying vec3 vNormal;",
            "varying vec2 vUv;",
            "varying float vCurv;",
            "varying float vMedial;",
            "varying float vThickmix;",
            // "varying float vDrop;",

            "varying vec3 vPos_x[2];",
        "#ifdef TWOD",
            "varying vec3 vPos_y[2];",
        "#endif",

            utils.mixer(morphs),

            "void main() {",
                utils.thickmixer_main,
                "vThickmix = use_thickmix;",

                "vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );",
                "vViewPosition = -mvPosition.xyz;",

                //Find voxel positions with both transforms (2D colormap x and y datasets)
                "vPos_x[0] = (volxfm[0]*vec4(position,1.)).xyz;",
            "#ifdef TWOD",
                "vPos_y[0] = (volxfm[1]*vec4(position,1.)).xyz;",
            "#endif",
        "#ifdef CORTSHEET",
                "vPos_x[1] = (volxfm[0]*vec4(wm.xyz,1.)).xyz;",
            "#ifdef TWOD",
                "vPos_y[1] = (volxfm[1]*vec4(wm.xyz,1.)).xyz;",
            "#endif",
        "#endif",

        "#ifdef CORTSHEET",
                "vec3 mpos = mix(position, wm.xyz, use_thickmix);",
                "vec3 mnorm = mix(normal, wmnorm, use_thickmix);",
        "#else",
                "vec3 mpos = position;",
                "vec3 mnorm = normal;",
        "#endif",

                //Overlay
                "vUv = uv;",
                // "vDrop = dropout;",
                "vMedial = auxdat.x;",
                "vCurv = auxdat.y;",

                "vec3 pos, norm;",
                "mixfunc(mpos, mnorm, pos, norm);",

                // "norm = mix(flatBumpNorms, normalize(onorm), thickmix);",
                // "norm = normalize(flatBumpNorms);",

            "#ifdef CORTSHEET",
                // 
                "#ifdef HASFLAT",
                    "pos += clamp(surfmix*"+(morphs-1)+"., 0., 1.) * normalize(norm) * mix(1., 0., use_thickmix) * flatheight * f_bumpyflat;",
                "#else",
                    "pos += clamp(surfmix*"+(morphs-1)+"., 0., 1.) * normalize(norm) * .62 * distance(position, wm.xyz) * mix(1., 0., use_thickmix);",
                "#endif",
            "#endif",

                "#ifdef HASFLAT",
                    "vNormal = normalMatrix * mix(norm, flatBumpNorms, (1.0 - use_thickmix) * clamp(surfmix*"+(morphs-1)+". - 1.0, 0., 1.) * f_bumpyflat);",
                "#else",
                    "vNormal = normalMatrix * norm;",
                "#endif",

                "gl_Position = projectionMatrix * modelViewMatrix * vec4( pos, 1.0 );",

            "}"
            ].join("\n");

            var fragHead = [
            "#extension GL_OES_standard_derivatives: enable",
            //"#extension GL_OES_texture_float: enable",

            THREE.ShaderChunk[ "lights_phong_pars_fragment" ],
        
        "varying vec2 vUv;",
        "#ifdef ROI_RENDER",
            "uniform sampler2D overlay;",
        "#endif",
        "#ifdef EXTRATEX",
            "uniform sampler2D extratex;",
        "#endif",

            "uniform float thickmix;",
            // utils.thickmixer,
            "uniform float brightness;",
            "uniform float smoothness;",
            "uniform float contrast;",
            "uniform vec3 voxlineColor;",
            "uniform float voxlineWidth;",

            "uniform sampler2D colormap;",
            "uniform float vmin[2];",
            "uniform float vmax[2];",
            "uniform float framemix;",
            "uniform vec2 mosaic[2];",
            "uniform vec2 dshape[2];",
            "uniform sampler2D data[4];",

            // "uniform float hatchAlpha;",
            // "uniform vec3 hatchColor;",
            // "uniform sampler2D hatch;",
            // "uniform vec2 hatchrep;",
            // "varying float vDrop;",

            "varying vec3 vPos_x[2];",
            "varying vec3 vPos_y[2];",

            "varying float vCurv;",
            "varying float vMedial;",
            "varying float vThickmix;",
            
            utils.standard_frag_vars,
            utils.rand,
            utils.edge,
            utils.colormap,
            utils.pack,
            utils.samplers,

            "void main() {",
                //Curvature Underlay
                "float ctmp = clamp(vCurv / smoothness, -0.5, 0.5);", // use limits here too
                "float curv = clamp(ctmp * contrast + brightness, 0.0, 1.0);",
                "vec4 cColor = vec4(vec3(curv), 1.0);", 

                "vec3 coord_x, coord_y;",
            "#ifdef RGBCOLORS",
                "vec4 color[2]; color[0] = vec4(0.), color[1] = vec4(0.);",
            "#else",
                "vec4 values = vec4(0.);",
            "#endif",
                "",
            ].join("\n");

            //Create samplers for texture volume sampling
            var fragMid = "";      
            var factor = layers > 1 ? (1/layers).toFixed(6) : "1.";
            var sampling = [
        "#ifdef RGBCOLORS",
                "color[0] += "+factor+"*"+sampler+"_x(data[0], coord_x);",
                "color[1] += "+factor+"*"+sampler+"_x(data[1], coord_x);",
        "#else",
                "values.x += "+factor+"*"+sampler+"_x(data[0], coord_x).r;",
                "values.y += "+factor+"*"+sampler+"_x(data[1], coord_x).r;",
            "#ifdef TWOD",
                "values.z += "+factor+"*"+sampler+"_y(data[2], coord_y).r;",
                "values.w += "+factor+"*"+sampler+"_y(data[3], coord_y).r;",
            "#endif",
        "#endif",
            ].join("\n");

            if (volume == 0) {
                fragMid += [
                    "coord_x = vPos_x[0];",
                "#ifdef TWOD",
                    "coord_y = vPos_y[0];",
                "#endif",
                    sampling,
                    ""
                ].join("\n");
            } else if (volume == 1) {
                if (layers == 1 && !dither) {
                    fragMid += [
                        "coord_x = mix(vPos_x[0], vPos_x[1], vThickmix);",
                    "#ifdef TWOD",
                        "coord_y = mix(vPos_y[0], vPos_y[1], vThickmix);",
                    "#endif",
                        sampling,
                        "",
                    ].join("\n");
                }
                else {
                    var sample_depth;
                    var step = 1 / (layers - 1);
                    fragMid += "vec2 rseed;\nfloat randval;\n";
                    for (var i = 0; i < layers; i++) {
                        if (dither) {
                            fragMid += [
                                "rseed = gl_FragCoord.xy + vec2(2.34*"+i.toFixed(3)+", 3.14*"+i.toFixed(3)+");",
                                "randval = rand(rseed);"].join("\n");
                            sample_depth = "randval";
                        } else {
                            sample_depth = (step * i).toFixed(3);
                        }
                        fragMid += [
                            "coord_x = mix(vPos_x[0], vPos_x[1], "+sample_depth+");",
                        "#ifdef TWOD",
                            "coord_y = mix(vPos_y[0], vPos_y[1], "+sample_depth+");",
                        "#endif",
                            sampling,
                            "", 
                        ].join("\n");
                    }
                }
            }

            var fragTail = [
    "#ifdef HALO_RENDER",
                "if (vMedial < .999) {",
                    "float dweight = gl_FragCoord.w;",
                    "float value = dweight * vnorm(values).x;",

                    "gl_FragColor.rg = encode_float_vec2(value);",
                    "gl_FragColor.ba = encode_float_vec2(dweight);",
                    //"gl_FragColor = vec4(res / 256., 1. / 256.);",
                    //"gl_FragColor = vec4(vec3(gl_FragCoord.w), 1.);",
                "} else if (surfmix > "+((morphs-2)/(morphs-1))+") {",
                    "gl_FragColor = vec4(0.);",
                "}",
    "#else",
            "#ifdef RGBCOLORS",
                "vec4 vColor = mix(color[0], color[1], framemix);",
            "#else",
                "vec4 vColor = colorlut(values);",
            "#endif",
                "vColor *= dataAlpha;",
                //"vColor.a = (values.x - vmin[0]) / (vmax[0] - vmin[0]);",

        "#ifdef VOXLINE",
            "#ifdef CORTSHEET",
                "vec3 coord = mix(vPos_x[0], vPos_x[1], vThickmix);",
            "#else",
                "vec3 coord = vPos_x[0];",
            "#endif",
                "vec3 edge = abs(fract(coord) - vec3(0.5));",
                "vColor = mix(vec4(voxlineColor, 1.), vColor, edgeFactor(edge*1.001));",
        "#endif",

                //Cross hatch / dropout layer
                // "float hw = gl_FrontFacing ? hatchAlpha*vDrop : 1.;",
                // "vec4 hColor = hw * vec4(hatchColor, 1.) * texture2D(hatch, vUv*hatchrep);",

                //roi layer
            "#ifdef ROI_RENDER",
                "vec4 rColor = (1. - step(.001, vMedial)) * texture2D(overlay, vUv);",
            "#endif",
            "#ifdef EXTRATEX",
                "vec4 tColor = (1. - step(.001, vMedial)) * texture2D(extratex, vUv);",
            "#endif",

                "gl_FragColor = cColor;",
                "gl_FragColor = vColor + (1.-vColor.a)*gl_FragColor;",
                // "gl_FragColor = hColor + (1.-hColor.a)*gl_FragColor;",
            "#ifdef ROI_RENDER",
                "gl_FragColor = rColor + (1.-rColor.a)*gl_FragColor;",
            "#endif",
            "#ifdef EXTRATEX",
                "gl_FragColor = tColor + (1.-tColor.a)*gl_FragColor;",
            "#endif",
                THREE.ShaderChunk[ "lights_phong_fragment" ],
    "#endif",
            "}"
            ].join("\n");

            var attributes = {
                wm: { type: 'v4', value:null },
                wmnorm: { type: 'v3', value:null },
                auxdat: { type: 'v4', value:null },
                wmarea: { type: 'f', value:null },
                pialarea: { type: 'f', value:null },
                // flatBumpNorms: { type: 'v3', value:null },
                // flatheight: { type: 'f', value:null },
            };
            if (opts.hasflat) {
                attributes.flatBumpNorms = { type: 'v3', value:null };
                attributes.flatheight = { type: 'f', value:null };
            }
            for (var i = 0; i < morphs-1; i++) {
                attributes['mixSurfs'+i] = { type:'v4', value:null};
                attributes['mixNorms'+i] = { type:'v3', value:null};
            }

            return {vertex:header+vertShade, fragment:header+fragHead+fragMid+fragTail, attrs:attributes};
        },

        surface_vertex: function(opts) {
            var header = "";
            if (opts.rgb)
                header += "#define RGBCOLORS\n";
            if (opts.twod)
                header += "#define TWOD\n";

            var morphs = opts.morphs;
            var volume = opts.volume || 0;
            if (volume > 0)
                header += "#define CORTSHEET\n";
            if (opts.rois)
                header += "#define ROI_RENDER\n";
            if (opts.extratex)
                header += "#define EXTRATEX\n";
            if (opts.equivolume)
                header += "#define EQUIVOLUME\n"

            var vertShade =  [
            THREE.ShaderChunk[ "lights_phong_pars_vertex" ],
            "uniform sampler2D colormap;",
            "uniform float vmin[2];",
            "uniform float vmax[2];",
            "uniform float framemix;",
            // "uniform float thickmix;",
            utils.thickmixer,

            "varying vec4 vColor;",
    "#ifdef RGBCOLORS",
            "attribute vec4 data0;",
            "attribute vec4 data1;",
    "#else",
            "attribute float data0;",
            "attribute float data1;",
            "attribute float data2;",
            "attribute float data3;",
    "#endif",

            "attribute vec4 wm;",
            "attribute vec3 wmnorm;",
            "attribute vec4 auxdat;",
            // "attribute float dropout;",
            
            "varying vec3 vViewPosition;",
            "varying vec3 vNormal;",
            "varying vec2 vUv;",
            "varying float vCurv;",
            "varying float vMedial;",
            // "varying float vDrop;",

            utils.mixer(morphs),

            "void main() {",

                utils.thickmixer_main,

                "vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );",
                "vViewPosition = -mvPosition.xyz;",

        "#ifdef RGBCOLORS",
                "vColor = mix(data0, data1, framemix);",
        "#else",
                "vec2 cuv;",
        //         "vValue.x = (mix(data0, data1, framemix) - vmin[0]) / (vmax[0] - vmin[0]);",
                "cuv.x = (mix(data0, data1, framemix) - vmin[0]) / (vmax[0] - vmin[0]);",
            "#ifdef TWOD",
        //         "vValue.y = (mix(data2, data3, framemix) - vmin[1]) / (vmax[1] - vmin[1]);",
                "cuv.y = (mix(data2, data3, framemix) - vmin[1]) / (vmax[1] - vmin[1]);",
            "#endif",
                "vColor = texture2D(colormap, cuv);",
        "#endif",

        "#ifdef CORTSHEET",
                "vec3 mpos = mix(position, wm.xyz, use_thickmix);",
                "vec3 mnorm = mix(normal, wmnorm, use_thickmix);",
        "#else",
                "vec3 mpos = position;",
                "vec3 mnorm = normal;",
        "#endif",

                //Overlay
                "vUv = uv;",
                // "vDrop = dropout;",
                "vMedial = auxdat.x;",
                "vCurv = auxdat.y;",

                "vec3 pos, norm;",
                "mixfunc(mpos, mnorm, pos, norm);",

            "#ifdef CORTSHEET",
                "pos += clamp(surfmix*"+(morphs-1)+"., 0., 1.) * normalize(norm) * .62 * distance(position, wm.xyz) * mix(1., 0., use_thickmix);",
            "#endif",

                "vNormal = normalMatrix * norm;",
                "gl_Position = projectionMatrix * modelViewMatrix * vec4( pos, 1.0 );",

            "}"
            ].join("\n");

            var fragShade = [
            "#extension GL_OES_standard_derivatives: enable",
            //"#extension GL_OES_texture_float: enable",

            THREE.ShaderChunk[ "lights_phong_pars_fragment" ],

        "varying vec2 vUv;",
        "#ifdef ROI_RENDER",
            "uniform sampler2D overlay;",
        "#endif",
        "#ifdef EXTRATEX",
            "uniform sampler2D extratex;",
        "#endif",

            "uniform float brightness;",
            "uniform float smoothness;",
            "uniform float contrast;",

            // "uniform float hatchAlpha;",
            // "uniform vec3 hatchColor;",
            // "uniform sampler2D hatch;",
            // "uniform vec2 hatchrep;",
            // "varying float vDrop;",
            "varying float vCurv;",
            "varying float vMedial;",
            "uniform float thickmix;",
            // utils.thickmixer,

            utils.standard_frag_vars,

        // "#ifdef RGBCOLORS",
            "varying vec4 vColor;",
        // "#else",
        //     "varying vec2 vValue;",
        // "#endif",

            "void main() {",
                //Curvature Underlay
                "float ctmp = clamp(vCurv / smoothness, -0.5, 0.5);", // use limits here too
                "float curv = clamp(ctmp * contrast + brightness, 0.0, 1.0);",
                //"vec4 cColor = vec4(vec3(curv) * brightness, brightness);", // Old way. Important to actually have alpha?
                "vec4 cColor = vec4(vec3(curv), 1.0);",

                //Cross hatch / dropout layer
                // "float hw = gl_FrontFacing ? hatchAlpha*vDrop : 1.;",
                // "vec4 hColor = hw * vec4(hatchColor, 1.) * texture2D(hatch, vUv*hatchrep);",

                //roi layer
            "#ifdef ROI_RENDER",
                "vec4 rColor = (1. - step(.001, vMedial)) * texture2D(overlay, vUv);",
            "#endif",
            "#ifdef EXTRATEX",
                "vec4 tColor = (1. - step(.001, vMedial)) * texture2D(extratex, vUv);",
            "#endif",

            // "#ifndef RGBCOLORS",
            //     "vec4 vColor = texture2D(colormap, vValue);",
            // "#endif",


                "gl_FragColor = cColor;",
                "gl_FragColor = vColor + (1.-vColor.a)*gl_FragColor;",
                //"gl_FragColor = vec4(1., 0., 0., 1.);",
                // "gl_FragColor = hColor + (1.-hColor.a)*gl_FragColor;",
            "#ifdef ROI_RENDER",
                "gl_FragColor = rColor + (1.-rColor.a)*gl_FragColor;",
            "#endif",
            "#ifdef EXTRATEX",
                "gl_FragColor = tColor + (1.-tColor.a)*gl_FragColor;",
            "#endif",
                THREE.ShaderChunk[ "lights_phong_fragment" ],
            "}"
            ].join("\n");

            var attributes = {
                wm: { type: 'v4', value:null },
                wmnorm: { type: 'v3', value:null },
                auxdat: { type: 'v4', value:null },
                wmarea: { type: 'f', value:null },
                pialarea: { type: 'f', value:null },
            };

            for (var i = 0; i < 4; i++)
                attributes['data'+i] = {type:opts.rgb ? 'v4':'f', value:null};

            for (var i = 0; i < morphs-1; i++) {
                attributes['mixSurfs'+i] = { type:'v4', value:null };
                attributes['mixNorms'+i] = { type:'v3', value:null };
            }

            return {vertex:header+vertShade, fragment:header+fragShade, attrs:attributes};
        },

        cmap_quad: function() {
            //Colormaps the full-screen quad, used for stage 2 of volume integration
            var vertShade = [
                "void main() {",
                    "gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
                "}",
            ].join("\n");
            var fragShade = [
                utils.pack,
                "uniform vec2 screen_size;",
                "uniform sampler2D screen;",
                "uniform sampler2D colormap;",
                "void main() {",
                    "vec4 data = texture2D(screen, gl_FragCoord.xy / screen_size);",
                    "float value = decode_float_vec2(data.rg);",
                    "float raw = value / decode_float_vec2(data.ba);",
                    //"if (value.a > 0.) {",
                       "gl_FragColor = texture2D(colormap, vec2(raw, 0.));",
                    //"} else {",
                    //   "discard;",
                    //"}",
                    //"gl_FragColor = vec4(vec3(raw), 1.);",
                    //"gl_FragColor = vec4(value.rgb, 1.);",
                "}",
            ].join("\n");
            return {vertex:vertShade, fragment:fragShade, attrs:{}};
        },
        
        pick: function(opts) {
            var header = "";
            var morphs = opts.morphs;
            if (opts.volume > 0)
                header += "#define CORTSHEET\n";

            var vertShade = [
                // "uniform float thickmix;",
                utils.thickmixer,

                utils.mixer(morphs),
                "attribute vec4 wm;",
                "attribute vec3 wmnorm;",
                "attribute vec4 auxdat;",

                "varying vec3 vPos;",

                "void main() {",
                    utils.thickmixer_main,

            "#ifdef CORTSHEET",
                    "vec3 mpos = mix(position, wm.xyz, use_thickmix);",
                    "vec3 mnorm = mix(normal, wmnorm, use_thickmix);",
                    "vPos = mix(position, wm.xyz, 0.5);",
            "#else",
                    "vec3 mpos = position;",
                    "vec3 mnorm = normal;",
                    "vPos = position;",
            "#endif",

                    "vec3 pos, norm;",
                    "mixfunc(mpos, mnorm, pos, norm);",

                "#ifdef CORTSHEET",
                    "pos += clamp(surfmix*"+(morphs-1)+"., 0., 1.) * normalize(norm) * .62 * distance(position, wm.xyz) * mix(1., 0., use_thickmix);",
                "#endif",

                    "gl_Position = projectionMatrix * modelViewMatrix * vec4( pos, 1.0 );",
                "}",
            ].join("\n");

            var fragShades = [];
            var dims = ['x', 'y', 'z'];
            for (var i = 0; i < 3; i++){
                var dim = dims[i];
                var shade = [
                "uniform vec3 min;",
                "uniform vec3 max;",

                "varying vec3 vPos;",
                utils.pack,
                "void main() {",
                    "float norm = (vPos."+dim+" - min."+dim+") / (max."+dim+" - min."+dim+");", 
                    "gl_FragColor = pack_float(norm);",
                "}"
                ].join("\n");
                fragShades.push(header+shade);
            }

            var attributes = {
                wm: { type: 'v4', value:null },
                wmnorm: { type: 'v3', value:null },
                auxdat: { type: 'v4', value:null },
            };

            for (var i = 0; i < morphs-1; i++) {
                attributes['mixSurfs'+i] = { type:'v4', value:null};
                attributes['mixNorms'+i] = { type:'v3', value:null};
            }

            return {vertex:header+vertShade, fragment:fragShades, attrs:attributes};
        },

        depth: function(opts) {
            var header = "";
            var morphs = opts.morphs;
            if (opts.volume > 0)
                header += "#define CORTSHEET\n";

            var vertShade = [
                "uniform float thickmix;",

                utils.mixer(morphs),
                "attribute vec4 wm;",
                "attribute vec3 wmnorm;",

                "void main() {",
            "#ifdef CORTSHEET",
                    "vec3 mpos = mix(position, wm.xyz, thickmix);",
                    "vec3 mnorm = mix(normal, wmnorm, thickmix);",
            "#else",
                    "vec3 mpos = position;",
                    "vec3 mnorm = normal;",
            "#endif",

                    "vec3 pos, norm;",
                    "mixfunc(mpos, mnorm, pos, norm);",

                "#ifdef CORTSHEET",
                    "pos += clamp(surfmix*"+(morphs-1)+"., 0., 1.) * normalize(norm) * .62 * distance(position, wm.xyz) * mix(1., 0., thickmix);",
                "#endif",

                    "gl_Position = projectionMatrix * modelViewMatrix * vec4( pos, 1.0 );",
                "}",
            ].join("\n");

            var fragShade = [
                "uniform vec3 min;",
                "uniform vec3 max;",
                
                utils.pack,
                "void main() {",
                    "gl_FragColor = pack_float(gl_FragCoord.z);",
                "}"
            ].join("\n");

            var attributes = {
                wm: { type: 'v4', value:null },
                wmnorm: { type: 'v3', value:null },
            };

            for (var i = 0; i < morphs-1; i++) {
                attributes['mixSurfs'+i] = { type:'v4', value:null};
                attributes['mixNorms'+i] = { type:'v3', value:null};
            }

            return {vertex:header+vertShade, fragment:fragShade, attrs:attributes};
        },
    };

    return module;
})();

var Shaders = new Shaderlib();
