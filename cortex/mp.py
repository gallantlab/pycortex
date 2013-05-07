import sys
import marshal
import multiprocessing as mp
try:
    import progressbar as pb
except ImportError:
    pass

def map(func, iterable, procs = mp.cpu_count()):
    input, output = mp.Queue(), mp.Queue()
    length = mp.Value('i',0)
    
    def _fill(iterable, procs, input, output):
        for data in enumerate(iterable):
            input.put(data)
            length.value += 1
        for _ in range(procs*2):
            input.put((-1,-1))
        
    def _func(proc, input, output):
        idx, data = input.get()
        while idx != -1:
            output.put((idx, func(data)))
            idx, data = input.get()
    
    filler = mp.Process(target = _fill, args=(iterable, procs, input, output))
    filler.daemon = True
    filler.start()
    for i in range(procs):
        proc = mp.Process(target=_func, args=(i, input, output))
        proc.daemon = True
        proc.start()
    
    try:
        iterlen = len(iterable)
    except:
        filler.join()
        iterlen = length.value

    data = [[]]*iterlen
    try:
        progress = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar()], maxval=iterlen)
        progress.start()
        for i in xrange(iterlen):
            idx, result = output.get()
            data[idx] = result
            progress.update(i+1)
        progress.finish()
    except NameError:
        for _ in xrange(iterlen):
            idx, result = output.get()
            data[idx] = result
        
    return data

if __name__ == "__main__":
    #pool = Pool()
    #data = pool.map(
    map(lambda x: max(x), zip(*(iter(xrange(65536)),)*3))
