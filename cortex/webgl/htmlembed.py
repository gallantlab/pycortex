import os
import re
import binascii
import mimetypes

import html5lib

from . import serve

def _resolve_path(filename, roots):
    for root in roots:
        p = os.path.join(root, filename)
        if os.path.exists(p):
            return p
    else:
        raise IOError("Path '%s' doesn't exist under any root dir (%s)" % (filename, roots))

def _embed_css(cssfile, rootdirs):
    csspath, fname = os.path.split(cssfile)
    with open(cssfile) as fp:
        css = fp.read()
        cssparse = re.compile(r'(.*?){([^}]+)}', re.S)
        urlparse = re.compile(r'url\(([^\)]+)\)')
        cssout = []
        for selector, content in cssparse.findall(css):
            for url in urlparse.findall(content):
                if not url.strip().startswith('data:'):
                    imgpath = _resolve_path(os.path.join(csspath, url), rootdirs)
                    content = re.sub(url, serve.make_base64(imgpath), content)

            cssout.append("%s {\n%s\n}"%(selector, content))
        return '\n'.join(cssout)

def _embed_js(dom, script, rootdirs):
    wparse = re.compile(r"new Worker\(\s*(['\"].*?['\"])\s*\)", re.S)
    aparse = re.compile(r"attr\(\s*['\"]src['\"]\s*,\s*['\"](.*?)['\"]\)")
    with open(_resolve_path(script.getAttribute("src"), rootdirs)) as jsfile:
        jssrc = jsfile.read()
        for worker in wparse.findall(jssrc):
            wid = os.path.splitext(os.path.split(worker.strip('"\''))[1])[0]
            wtext = _embed_worker(_resolve_path(worker.strip('"\''), rootdirs))
            wscript = dom.createElement("script")
            wscript.setAttribute("type", "text/js-worker")
            wscript.setAttribute("id", wid)
            wscript.appendChild(dom.createTextNode(wtext))
            script.parentNode.insertBefore(wscript, script)
            rplc = "window.URL.createObjectURL(new Blob([document.getElementById('%s').textContent]))"%wid
            jssrc = jssrc.replace(worker, rplc)

        for src in aparse.findall(jssrc):
            if not src.strip().startswith("data:"):
                jspath = _resolve_path(src.strip('\'"'), rootdirs)
                jssrc = jssrc.replace(src, "%s"%serve.make_base64(jspath))

        script.removeAttribute("src")
        script.appendChild(dom.createTextNode(jssrc.decode('utf-8')))

def _embed_worker(worker):
    wparse = re.compile(r"importScripts\((.*)\)")
    wpath = os.path.split(worker)[0]
    with open(worker) as wfile:
        wdata = wfile.read()
        for simport in wparse.findall(wdata):
            imports = []
            for fname in simport.split(','):
                iname = os.path.join(wpath, fname.strip('\'" '))
                with open(iname) as fp:
                    imports.append(fp.read())
            wdata = wdata.replace("importScripts(%s)"%simport, '\n'.join(imports))
        return wdata

def _embed_images(dom, rootdirs):
    for img in dom.getElementsByTagName("img"):
        imgfile = _resolve_path(img.getAttribute("src"), rootdirs)
        img.setAttribute("src", serve.make_base64(imgfile))

def embed(rawhtml, outfile, rootdirs=(serve.cwd,)):
    parser = html5lib.HTMLParser(tree=html5lib.treebuilders.getTreeBuilder("dom"))
    dom = parser.parse(rawhtml)
    head = dom.getElementsByTagName("head")[0]
    wurl = dom.createElement("script")
    wurl.setAttribute("type", "text/javascript")
    wurl.appendChild(dom.createTextNode('''
if (window.webkitURL)
    window.URL = window.webkitURL;
'''))
    head.insertBefore(wurl, head.childNodes[0])

    for script in dom.getElementsByTagName("script"):
        stype = script.getAttribute("type")
        if stype == "text/html":
            sdom = parser.parse(script.childNodes[0].wholeText)
            _embed_images(sdom, rootdirs)
            while len(script.childNodes) > 0:
                for c in script.childNodes:
                    script.removeChild(c)

            shtml = ""
            for el in sdom.getElementsByTagName("body")[0].childNodes:
                shtml += el.toxml()
            script.appendChild(dom.createTextNode(shtml))
        elif stype == "text/javascript":
            src = script.getAttribute("src")
            if len(src) > 0:
                _embed_js(dom, script, rootdirs)
    
    for css in dom.getElementsByTagName("link"):
        if (css.getAttribute("type") == "text/css"):
            csstext = _embed_css(_resolve_path(css.getAttribute("href"), rootdirs), rootdirs)
            ncss = dom.createElement("style")
            ncss.setAttribute("type", "text/css")
            ncss.appendChild(dom.createTextNode(csstext))
            css.parentNode.insertBefore(ncss, css)
            css.parentNode.removeChild(css)

    _embed_images(dom, rootdirs)

    #Save out the new html file
    with open(outfile, "w") as htmlfile:
        serializer = html5lib.serializer.htmlserializer.HTMLSerializer()
        walker = html5lib.treewalkers.getTreeWalker("dom")

        for line in serializer.serialize(walker(dom)):
            htmlfile.write(line.encode("utf-8"))
