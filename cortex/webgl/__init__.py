
def show(*args, **kwargs):
    from . import view
    return view.show(*args, **kwargs)

def make_static(*args, **kwargs):
    from . import view
    return view.make_static(*args, **kwargs)