import numpy 

def reduce_arr(arr:list,newlen:int):
    if not arr:
        return []
    
    newlen      = max(1,newlen)
    arr         = numpy.asarray(arr)
    factor      = len(arr) // newlen
    reduced     = arr[-newlen*factor:].reshape(newlen, factor).mean(axis=1)
   
    return reduced