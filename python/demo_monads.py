from typing import List, Dict

def square(x: int) -> Dict[str, object]:
    return {
        "result": x * x,
        "logs": [f"Squared {x} to get {x * x}."]
    }

def add_one(x: Dict[str, object]) -> Dict[str, object]:
    result = x["result"] + 1
    logs: List[str] = x["logs"] + [f"Added 1 to {x['result']} to get {result}."]
    return {"result": result, "logs": logs}

out = add_one(square(2))
print(out)
