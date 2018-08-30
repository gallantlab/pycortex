

class IntrinsicDelaunaySurface():
    def __init__(self, surface, new_polys=None, new_laplace_beltrami=None):
        self = super(surface.pts, new_polys)

        if new_laplace_beltrami is None:
            # need to set setter
            self.laplace_beltrami = new_laplace_beltrami

        self.idt_raw_surface = surface

        # need to set edge lengths and edge angles as well

        # check validity of idt surface

    def is_intrinsic_delaunay(self):
        return (self.cotangent_weights < 0).sum() == 0
