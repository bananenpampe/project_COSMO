import itertools


def grouper(n, iterable):
    """Helper function that yields an iterable in chunks of n
    """
    #from https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk