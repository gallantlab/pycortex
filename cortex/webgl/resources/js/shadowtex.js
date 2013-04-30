var vshade = [
//"uniform vec4 color;",
"varying vec2 vUv;",
"void main(void) {",
    "gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
    "vUv = uv;",
"}",
].join("\n");

var fshadeh = [
"uniform sampler2D tex;", // the texture with the scene you want to blur
//"uniform vec4 color;",
"varying vec2 vUv;", 
 
"uniform float blur;",
 
"void main(void) {",

"   vec4 sum = vec4(0.0);",
   // blur in y (vertical)
   // take nine samples, with the distance blur between them
"   sum += texture2D(tex, vec2(vUv.x - 4.0*blur, vUv.y)) * 0.05;",
"   sum += texture2D(tex, vec2(vUv.x - 3.0*blur, vUv.y)) * 0.09;",
"   sum += texture2D(tex, vec2(vUv.x - 2.0*blur, vUv.y)) * 0.12;",
"   sum += texture2D(tex, vec2(vUv.x - blur, vUv.y)) * 0.15;",
"   sum += texture2D(tex, vec2(vUv.x, vUv.y)) * 0.16;",
"   sum += texture2D(tex, vec2(vUv.x + blur, vUv.y)) * 0.15;",
"   sum += texture2D(tex, vec2(vUv.x + 2.0*blur, vUv.y)) * 0.12;",
"   sum += texture2D(tex, vec2(vUv.x + 3.0*blur, vUv.y)) * 0.09;",
"   sum += texture2D(tex, vec2(vUv.x + 4.0*blur, vUv.y)) * 0.05;",
   
"   gl_FragColor = sum;",
//"   gl_FragColor = vec4(vUv.x, vUv.y, 0., 1.);",
"}",
].join("\n");

var fshadev = [
"uniform sampler2D tex;", // this should hold the texture rendered by the horizontal blur pass
//"uniform vec4 color;",
"varying vec2 vUv;", 
 
"uniform float blur;",
 
"void main(void) {",
   "vec4 sum = vec4(0.0);",
 
   // blur in y (vertical)
   // take nine samples, with the distance blur between them
   "sum += texture2D(tex, vec2(vUv.x, vUv.y - 4.0*blur)) * 0.05;",
   "sum += texture2D(tex, vec2(vUv.x, vUv.y - 3.0*blur)) * 0.09;",
   "sum += texture2D(tex, vec2(vUv.x, vUv.y - 2.0*blur)) * 0.12;",
   "sum += texture2D(tex, vec2(vUv.x, vUv.y - blur)) * 0.15;",
   "sum += texture2D(tex, vec2(vUv.x, vUv.y)) * 0.16;",
   "sum += texture2D(tex, vec2(vUv.x, vUv.y + blur)) * 0.15;",
   "sum += texture2D(tex, vec2(vUv.x, vUv.y + 2.0*blur)) * 0.12;",
   "sum += texture2D(tex, vec2(vUv.x, vUv.y + 3.0*blur)) * 0.09;",
   "sum += texture2D(tex, vec2(vUv.x, vUv.y + 4.0*blur)) * 0.05;",
 
   "gl_FragColor = sum;",
"}",
].join("\n");

var composite = [
"uniform sampler2D over;",
"uniform sampler2D under;",
"varying vec2 vUv;", 

"void main(void) {",
    "vec4 o = texture2D(over, vUv);", 
    "vec4 u = texture2D(under, vUv);", 
    "gl_FragColor.rgb = o.rgb + u.rgb*(1.-o.a);",
    "gl_FragColor.a = o.a + u.a*(1.-o.a);",
"}",
].join("\n");

function ShadowTex(width, height, radius) {
    this.width = width;
    this.height = height;

    this.htex = new THREE.WebGLRenderTarget(width, height, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format:THREE.RGBAFormat,
        stencilBuffer:false,
    });
    this.vtex = new THREE.WebGLRenderTarget(width, height, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format:THREE.RGBAFormat,
        stencilBuffer:false,
    });
    this.hpass = new THREE.ShaderMaterial({
        vertexShader:vshade,
        fragmentShader:fshadeh,
        transparent:true,
        blending: THREE.CustomBlending,
        uniforms: {
            tex:{type:'t', value:0, texture:null}, 
            blur:{type:'f', value:radius / width}
        }
    })
    this.hpass.blendSrc = THREE.OneFactor;
    this.vpass = new THREE.ShaderMaterial({
        vertexShader:vshade,
        fragmentShader:fshadev,
        transparent:true,
        blending: THREE.CustomBlending,
        uniforms: {
            tex: {type:'t', value:0, texture:this.htex},
            blur:{type:'f', value:radius / height}
        }
    })
    this.vpass.blendSrc = THREE.OneFactor;

    this.opass = new THREE.ShaderMaterial({
        vertexShader:vshade,
        fragmentShader:composite,
        transparent:true,
        blending: THREE.CustomBlending,
        uniforms: {
            over: {type:'t', value:0, texture:null},
            under: {type:'t', value:1, texture:null}
        }
    });
    this.opass.blendSrc = THREE.OneFactor;
    
    this.fsquad = new THREE.Mesh(new THREE.PlaneGeometry(width, height), this.hpass);
    this.fsquad.rotation.x = Math.PI / 2;
    this.camera = new THREE.OrthographicCamera( width / - 2, width / 2, height / 2, height / - 2, -10000, 10000 );
    this.scene = new THREE.Scene();
    this.scene.add(this.camera);
    this.scene.add(this.fsquad);
}
ShadowTex.prototype = {
    setRadius: function(radius) {
        this.vpass.uniforms.blur.value = radius / this.height;
        this.hpass.uniforms.blur.value = radius / this.width;
    },
    blur: function(renderer, shadow) {
        var clearAlpha = renderer.getClearAlpha();
        var clearColor = renderer.getClearColor();
        renderer.setClearColorHex(0x0, 0);
        this.hpass.uniforms.tex.texture = this._setTex(shadow);
        this.fsquad.material = this.hpass;
        renderer.render(this.scene, this.camera, this.htex);
        this.vpass.uniforms.tex.texture = this.htex;
        this.fsquad.material = this.vpass;
        renderer.render(this.scene, this.camera, this.vtex);
        renderer.setClearColor(clearColor, clearAlpha);
        return this.vtex;
    },
    overlay: function(renderer, over) {
        var clearAlpha = renderer.getClearAlpha();
        var clearColor = renderer.getClearColor();
        renderer.setClearColorHex(0x0, 0);
        this.opass.uniforms.over.texture = this._setTex(over);
        this.opass.uniforms.under.texture = this.vtex;
        this.fsquad.material = this.opass;
        renderer.render(this.scene, this.camera, this.htex);
        renderer.setClearColor(clearColor, clearAlpha);
        return this.htex;
    }, 
    _setTex: function(tex) {
        tex.needsUpdate = true;
        tex.premultiplyAlpha = true;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        return tex;
    }
};