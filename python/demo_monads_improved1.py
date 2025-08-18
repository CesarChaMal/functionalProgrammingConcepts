from typing import Dict, List, Callable

def wrap_with_logs(x: int) -> Dict[str, object]:
    return {"result": x, "logs": []}

def run_with_logs(input_data: Dict[str, object], transform: Callable[[int], Dict[str, object]]) -> Dict[str, object]:
    new_number_with_logs = transform(input_data["result"])
    return {
        "result": new_number_with_logs["result"],
        "logs": input_data["logs"] + new_number_with_logs["logs"]
    }

def square(x: int) -> Dict[str, object]:
    return {
        "result": x * x,
        "logs": [f"Squared {x} to get {x * x}."]
    }

def add_one(x: int) -> Dict[str, object]:
    return {
        "result": x + 1,
        "logs": [f"Added 1 to {x} to get {x + 1}."]
    }

def multiply_by_three(x: int) -> Dict[str, object]:
    return {
        "result": x * 3,
        "logs": [f"Multiplied {x} by 3 to get {x * 3}."]
    }

a = wrap_with_logs(5)
b = run_with_logs(a, add_one)
c = run_with_logs(b, square)
d = run_with_logs(c, multiply_by_three)
print(d)