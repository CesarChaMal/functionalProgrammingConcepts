from dataclasses import dataclass
from typing import Callable, List

@dataclass
class NumberWithLogs:
    result: int
    logs: List[str]
    
    @classmethod
    def pure(cls, x: int) -> 'NumberWithLogs':
        return cls(x, [])
    
    @classmethod
    def of(cls, result: int, *logs: str) -> 'NumberWithLogs':
        return cls(result, list(logs))
    
    def flat_map(self, transform: Callable[[int], 'NumberWithLogs']) -> 'NumberWithLogs':
        next_value = transform(self.result)
        return NumberWithLogs(next_value.result, self.logs + next_value.logs)

def square(x: int) -> NumberWithLogs:
    return NumberWithLogs.of(x * x, f"Squared {x} to get {x * x}.")

def add_one(x: int) -> NumberWithLogs:
    return NumberWithLogs.of(x + 1, f"Added 1 to {x} to get {x + 1}.")

def multiply_by_three(x: int) -> NumberWithLogs:
    return NumberWithLogs.of(x * 3, f"Multiplied {x} by 3 to get {x * 3}.")

result = (NumberWithLogs.pure(5)
    .flat_map(add_one)
    .flat_map(square)
    .flat_map(multiply_by_three))

print(result)