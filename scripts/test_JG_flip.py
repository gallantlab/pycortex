import cortex

jgmask = cortex.get_cortical_mask("JG", "identity")
jgflipmask = cortex.get_cortical_mask("JG-flip", "identity")

anatfile = cortex.surfs.getAnat("JG")
