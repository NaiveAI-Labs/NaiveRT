"""JSON-safe dump of the Naive N0.5 model spec, hybrid spec and kernel geometry (the cache-layout contract)."""
import dataclasses
import types


def plain(o):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: plain(getattr(o, f.name)) for f in dataclasses.fields(o)}
    if isinstance(o, (dict, types.MappingProxyType)):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return repr(o)


def dump():
    from naivert.models.naive_n0p5 import model as m, kernel_geometry as g
    return {"spec": plain(m.NAIVE_N0P5_GEOMETRY), "hybrid": plain(m.NAIVE_N0P5_HYBRID),
            "geometry": {k: plain(getattr(g, k)) for k in dir(g) if k.isupper() and not k.startswith("_")}}


if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    json.dump(dump(), sys.stdout, indent=1, sort_keys=True)
