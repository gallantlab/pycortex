import os
import ConfigParser
import appdirs

defaults = dict(
    basic = dict(
        filestore=appdirs.user_data_dir("pycortex", "JamesGao"),
    ),
    mayavi_aligner = dict(
        line_width=1,
        point_size=5,
        outline_color='white',
        cmap='grey',
    ),
    webgl = dict(
    )
)

userdir = appdirs.user_data_dir("pycortex", "JamesGao")
usercfg = os.path.join(userdir, "options.cfg")

config = ConfigParser.ConfigParser(defaults)
found = config.read(['site.cfg', usercfg])
if len(found) < 1:
    for section, opts in defaults.items():
        config.add_section(section)
        for opt, value in opts.items():
            config.set(section, opt, value)

    try:
        os.makedirs(userdir)
    except OSError:
        pass
        
    with open(usercfg, 'w') as fp:
        config.write(fp)