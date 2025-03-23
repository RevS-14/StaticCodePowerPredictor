from strictyaml import Map, Int, Any, Bool, Any, Str

yaml_schema = Map({
    "loop": Map({
        "iters": Int(),
        "repeat_count": Any(),
        "returns": Str()
    }),
    "if": Map({
        "repeat_count": Any(),
        "returns": Str()
    }),
    "recursion": Map({
        "loops": Int(),
        "repeat_count": Any(),
        "returns": Str()
    }),
    "memory": Map({
        "size": Any(),
        "repeat_count": Any(),
        "returns": Str()
    }),
    "bitwise": Map({
        "repeat_count": Any(),
        "returns": Str()
    }),
    "pointerArithmetic": Map({
        "repeat_count": Any(),
        "returns": Str()
    }),
    "struct": Map({
        "repeat_count": Any(),
        "returns": Str()
    }),
    "functionPointer": Map({
        "repeat_count": Any(),
        "returns": Str()
    })
})