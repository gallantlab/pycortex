
def show(*args, **kwargs):
    from . import view
    view.show(*args, **kwargs)

def make_static(*args, **kwargs):
    from . import view
    view.make_static(*args, **kwargs)