Transform formats
=================
Functional data, usually collected by an epi sequence, typically does not have the same scan parameters as the anatomical MPRAGE scan used to generate the surfaces. Additionally, fMRI sequences which are usually optimized for T2* have drastically different and larger distortions than a typical T1 anatomical sequence. While automatic algorithms exist to align these two scan types, they will sometimes fail spectacularly, especially if a partial volume slice prescription is necessary.

pycortex includes a tool based on mayavi_ to do manual **affine** alignments. Please see the :module:`align` module for more information. Alternatively, if an automatic algorithm works well enough, you can also commit your own transform to the database. Transforms in pycortex always go from **fiducial to functional** space. They have four variables associated:

    * **Subject** : name of the subject, must match the surfaces used to create the transform
    * **Name** : A unique identifier for this transform
    * **type** : The type of transform -- from fiducial to functional **magnet** space, or fiducial to **coord** inate space
    * **epifile** : the filename of the functional data that the fiducial is aligned to

Transforms always store the epifile in order to allow visual validation of alignment using the :module:`align` module.

.. _mayavi: http://docs.enthought.com/mayavi/mayavi/