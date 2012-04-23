import os
import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self, path):
        if path == '':
            self.write(open("mixer.html").read())
        elif os.path.isfile(path):
            self.write(open(path).read())
        else:
            self.write_error(404)

application = tornado.web.Application([
    (r"/(.*)", MainHandler),
])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()