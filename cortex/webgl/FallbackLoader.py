import os
from tornado import template

class FallbackLoader(template.BaseLoader):
    """Loads templates from one of multiple potential directories, falling back 
    to next if the template is not found in the first directory.
    """
    def __init__(self, root_directories, **kwargs):
        super(FallbackLoader, self).__init__(**kwargs)
        self.roots = [os.path.abspath(d) for d in root_directories]
    
    def resolve_path(self, name, parent_path=None):
        if parent_path and parent_path[0] not in ["<", "/"] and not name.startswith("/"):
            for root in self.roots:
                current_path = os.path.join(root, parent_path)
                file_dir = os.path.dirname(os.path.abspath(current_path))
                relative_path = os.path.abspath(os.path.join(file_dir, name))
                newname = relative_path[len(root)+1:]
                ## Check if path exists
                if os.path.exists(relative_path):
                    return newname
            else:
                raise Exception("Couldn't find template.")
        return name

    def _create_template(self, name):
        for root in self.roots:
            path = os.path.join(root, name)
            if os.path.exists(path):
                f = open(path, "rb")
                t = template.Template(f.read(), name=name, loader=self)
                f.close()
                return t
        else:
            raise Exception("Couldn't find template.")
            
