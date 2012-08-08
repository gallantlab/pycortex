import os
import re
import binascii
import mimetypes

import html5lib

import serve

def _make_base64(imgfile):
    with open(imgfile) as img:
        mtype = mimetypes.guess_type(imgfile)[0]
        data = binascii.b2a_base64(img.read())
        return "data:{mtype};base64,{data}".format(mtype=mtype, data=data)


def _embed_css(cssfile):
    csspath, fname = os.path.split(cssfile)
    with open(cssfile) as fp:
        css = fp.read()
        cssparse = re.compile(r'(.*?){(.*?)}', re.S)
        urlparse = re.compile(r'url\((.*?)\)')
        cssout = []
        for selector, content in cssparse.findall(css):
            lines = []
            for line in content.split(';'):
                if len(line.strip()) > 0:
                    attr, val = line.strip().split(':')
                    url = urlparse.search(val)
                    if url is not None:
                        imgfile = os.path.join(csspath, url.group(1))
                        imgdat = "url(%s)"%_make_base64(imgfile)
                        val = urlparse.sub(imgdat, val)
                    lines.append("%s:%s"%(attr, val))
            cssout.append("%s {\n%s;\n}"%(selector, ';\n'.join(lines)))
        return '\n'.join(cssout)

js_prefix = '''
var wurl;
if (window.URL)
    wurl = window.URL.createObjectURL;
else if (window.webkitURL)
    wurl = window.webkitURL.createObjectURL;
'''
def _embed_js(dom, script):
    assert script.hasAttribute("src")
    with open(os.path.join(serve.cwd, script.getAttribute("src"))) as jsfile:
        jssrc = jsfile.read()
        jsparse = re.compile(r"new Worker(['\"](.*?)['\"])")
        for worker in jsparse.findall(jssrc):
            with open(os.path.join(serve.cwd, worker)) as wfile:
                wid = os.path.splitext(os.path.split(worker)[1])[0]
                wscript = dom.createElement("script")
                wscript.setAttribute("type", "text/js-worker")
                wscript.setAttribute("id", wid)
                wscript.appendChild(dom.createTextNode(wfile.read()))
                script.parentNode.insertBefore(wscript, script)
                jssrc.replace(worker, "wurl(new Blob([document.getElementById('"+wid+"').textContent]))")

        script.removeAttribute("src")
        script.appendChild(dom.createTextNode(js_prefix + jssrc))


def embed(rawhtml, outfile):
    parser = html5lib.HTMLParser(tree=html5lib.treebuilders.getTreeBuilder("dom"))
    dom = parser.parse(rawhtml)
    for script in dom.getElementsByTagName("script"):
        src = script.getAttribute("src")
        if len(src) > 0:
            _embed_js(dom, script)            
    
    for css in dom.getElementsByTagName("link"):
        if (css.getAttribute("type") == "text/css"):
            csspath = os.path.join(serve.cwd, css.getAttribute("href"))
            ncss = dom.createElement("style")
            ncss.setAttribute("type", "text/css")
            ncss.appendChild(dom.createTextNode(_embed_css(csspath)))
            css.parentNode.insertBefore(ncss, css)
            css.parentNode.removeChild(css)

    for img in dom.getElementsByTagName("img"):
        imgfile = os.path.join(serve.cwd, img.getAttribute("src"))
        if os.path.exists(imgfile):
            img.setAttribute("src", _make_base64(imgfile))
        else:
            print "Cannot find image to embed: %s"%imgfile

    #Save out the new html file
    with open(outfile, "w") as htmlfile:
        serializer = html5lib.serializer.htmlserializer.HTMLSerializer()
        walker = html5lib.treewalkers.getTreeWalker("dom")

        for line in serializer.serialize(walker(dom)):
            htmlfile.write(line)