import kaiwu as kw
print("kaiwu:", getattr(kw, "__version__", "no __version__"))
print("has Binary:", hasattr(kw.qubo, "Binary"))
print("has ndarray:", hasattr(kw.qubo, "ndarray"))
#print("dir(qubo) contains:", [k for k in dir(kw.qubo) if "ndarray" in k.lower() or "binary" in k.lower()])
