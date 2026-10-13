from __future__ import annotations

import ast
import json
import math
import re
import statistics
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import sympy

from .retrieval import LocalVectorStore


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "search_knowledge_base",
        "description": (
            "Search the private local document collection. Use this before answering "
            "questions that may depend on the user's documents. Results include citation IDs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused semantic search query.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "description": "Number of chunks to retrieve.",
                },
            },
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_current_weather",
        "description": (
            "Get live current weather for a city using the read-only Open-Meteo API. "
            "Use only when the user asks for current weather."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and optional country, for example Berlin, Germany.",
                }
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "evaluate_math",
        "description": (
            "Safely evaluate or simplify an arithmetic or symbolic expression. "
            "Supports +, -, *, /, %, powers, parentheses, symbols, pi, e, and common functions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression such as (15/100)*240 or sqrt(2)+sin(pi/4).",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "symbolic_math",
        "description": "Solve an equation, differentiate an expression, or integrate it.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["solve", "differentiate", "integrate"],
                },
                "expression": {
                    "type": "string",
                    "description": "For solve, use an equation like x^2 - 4 = 0.",
                },
                "variable": {
                    "type": "string",
                    "description": "A single variable name such as x.",
                },
            },
            "required": ["operation", "expression", "variable"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "analyze_numbers",
        "description": "Calculate descriptive statistics for a list of numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 1,
                    "maxItems": 1000,
                },
                "operation": {
                    "type": "string",
                    "enum": ["summary", "mean", "median", "stdev", "sum"],
                },
            },
            "required": ["values", "operation"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "matrix_math",
        "description": "Calculate a determinant, inverse, transpose, or matrix product.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["determinant", "inverse", "transpose", "multiply"],
                },
                "matrix_a": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "minItems": 1,
                    "maxItems": 20,
                },
                "matrix_b": {
                    "type": ["array", "null"],
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Second matrix for multiplication; otherwise null.",
                },
            },
            "required": ["operation", "matrix_a", "matrix_b"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "convert_units",
        "description": "Convert common length, mass, time, and temperature units.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {
                    "type": "string",
                    "enum": ["mm", "cm", "m", "km", "in", "ft", "mi", "g", "kg", "lb", "s", "min", "h", "day", "C", "F", "K"],
                },
                "to_unit": {
                    "type": "string",
                    "enum": ["mm", "cm", "m", "km", "in", "ft", "mi", "g", "kg", "lb", "s", "min", "h", "day", "C", "F", "K"],
                },
            },
            "required": ["value", "from_unit", "to_unit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


ALLOWED_FUNCTIONS = {
    "abs": sympy.Abs,
    "cos": sympy.cos,
    "exp": sympy.exp,
    "log": sympy.log,
    "sin": sympy.sin,
    "sqrt": sympy.sqrt,
    "tan": sympy.tan,
}
ALLOWED_CONSTANTS = {"pi": sympy.pi, "e": sympy.E}


class SafeMathParser(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes = 0

    def visit(self, node: ast.AST) -> sympy.Expr:  # type: ignore[override]
        self.nodes += 1
        if self.nodes > 100:
            raise ValueError("Expression is too complex.")
        return super().visit(node)

    def generic_visit(self, node: ast.AST) -> sympy.Expr:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression) -> sympy.Expr:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> sympy.Expr:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed.")
        if abs(float(node.value)) > 1e100:
            raise ValueError("Numeric constant is too large.")
        return sympy.sympify(node.value)

    def visit_Name(self, node: ast.Name) -> sympy.Expr:
        if node.id in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[node.id]
        if not node.id.isidentifier() or len(node.id) > 32 or node.id.startswith("_"):
            raise ValueError("Invalid variable name.")
        return sympy.Symbol(node.id, real=True)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> sympy.Expr:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("Unsupported unary operator.")

    def visit_BinOp(self, node: ast.BinOp) -> sympy.Expr:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return sympy.Mod(left, right)
        if isinstance(node.op, ast.Pow):
            if right.is_number and abs(float(right)) > 100:
                raise ValueError("Exponent is too large.")
            return left**right
        raise ValueError("Unsupported binary operator.")

    def visit_Call(self, node: ast.Call) -> sympy.Expr:
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise ValueError("Function is not allowed.")
        if node.keywords or len(node.args) not in {1, 2}:
            raise ValueError("Invalid function arguments.")
        return ALLOWED_FUNCTIONS[node.func.id](*(self.visit(arg) for arg in node.args))


def parse_math(expression: str) -> sympy.Expr:
    if not expression.strip() or len(expression) > 500:
        raise ValueError("Expression must contain 1 to 500 characters.")
    source = expression.strip().replace("^", "**")
    source = re.sub(r"(?<=\d)(?=[A-Za-z_(])", "*", source)
    source = re.sub(r"(?<=\))(?=[A-Za-z_(\d])", "*", source)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid mathematical expression.") from exc
    return SafeMathParser().visit(tree)


def evaluate_math(expression: str) -> dict[str, Any]:
    result = sympy.simplify(parse_math(expression))
    output: dict[str, Any] = {"ok": True, "exact": sympy.sstr(result)}
    if not result.free_symbols:
        output["decimal"] = sympy.sstr(sympy.N(result, 12))
    return output


def symbolic_math(operation: str, expression: str, variable: str) -> dict[str, Any]:
    if not variable.isidentifier() or len(variable) > 32 or variable.startswith("_"):
        raise ValueError("Invalid variable name.")
    symbol = sympy.Symbol(variable, real=True)
    if operation == "solve":
        if expression.count("=") > 1:
            raise ValueError("Equation may contain at most one equals sign.")
        sides = expression.split("=", maxsplit=1)
        equation = parse_math(sides[0]) - (parse_math(sides[1]) if len(sides) == 2 else 0)
        solutions = sympy.solve(equation, symbol)
        return {"ok": True, "solutions": [sympy.sstr(item) for item in solutions]}
    parsed = parse_math(expression)
    if operation == "differentiate":
        result = sympy.diff(parsed, symbol)
    elif operation == "integrate":
        result = sympy.integrate(parsed, symbol)
    else:
        raise ValueError(f"Unsupported symbolic operation: {operation}")
    return {"ok": True, "result": sympy.sstr(result)}


def analyze_numbers(values: list[float], operation: str) -> dict[str, Any]:
    if not values or len(values) > 1000:
        raise ValueError("Provide between 1 and 1000 values.")
    if operation == "sum":
        return {"ok": True, "sum": math.fsum(values)}
    if operation == "mean":
        return {"ok": True, "mean": statistics.fmean(values)}
    if operation == "median":
        return {"ok": True, "median": statistics.median(values)}
    if operation == "stdev":
        if len(values) < 2:
            raise ValueError("Standard deviation needs at least two values.")
        return {"ok": True, "sample_stdev": statistics.stdev(values)}
    if operation == "summary":
        return {
            "ok": True,
            "count": len(values),
            "sum": math.fsum(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
        }
    raise ValueError(f"Unsupported statistics operation: {operation}")


def _matrix_rows(value: list[list[float]]) -> list[list[float]]:
    if not value or len(value) > 20 or any(not row or len(row) > 20 for row in value):
        raise ValueError("Matrices must be between 1x1 and 20x20.")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValueError("All matrix rows must have equal length.")
    return value


def matrix_math(
    operation: str,
    matrix_a: list[list[float]],
    matrix_b: list[list[float]] | None,
) -> dict[str, Any]:
    a = sympy.Matrix(_matrix_rows(matrix_a))
    if operation == "determinant":
        if not a.is_square:
            raise ValueError("Determinant requires a square matrix.")
        return {"ok": True, "result": sympy.sstr(a.det())}
    if operation == "inverse":
        if not a.is_square:
            raise ValueError("Inverse requires a square matrix.")
        result = a.inv()
    elif operation == "transpose":
        result = a.T
    elif operation == "multiply":
        if matrix_b is None:
            raise ValueError("Matrix multiplication requires matrix_b.")
        result = a * sympy.Matrix(_matrix_rows(matrix_b))
    else:
        raise ValueError(f"Unsupported matrix operation: {operation}")
    return {"ok": True, "result": [[sympy.sstr(cell) for cell in row] for row in result.tolist()]}


UNIT_FACTORS = {
    "length": {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0, "in": 0.0254, "ft": 0.3048, "mi": 1609.344},
    "mass": {"g": 0.001, "kg": 1.0, "lb": 0.45359237},
    "time": {"s": 1.0, "min": 60.0, "h": 3600.0, "day": 86400.0},
}


def convert_units(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    temperatures = {"C", "F", "K"}
    if from_unit in temperatures or to_unit in temperatures:
        if from_unit not in temperatures or to_unit not in temperatures:
            raise ValueError("Cannot convert temperature to a non-temperature unit.")
        celsius = value if from_unit == "C" else (value - 32) * 5 / 9 if from_unit == "F" else value - 273.15
        result = celsius if to_unit == "C" else celsius * 9 / 5 + 32 if to_unit == "F" else celsius + 273.15
        return {"ok": True, "value": result, "unit": to_unit}
    for factors in UNIT_FACTORS.values():
        if from_unit in factors and to_unit in factors:
            return {"ok": True, "value": value * factors[from_unit] / factors[to_unit], "unit": to_unit}
    raise ValueError("Units belong to different dimensions.")


def _get_json(base_url: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{base_url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "tool-rag-starter/1.0"})
    with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed trusted hosts
        return json.loads(response.read().decode("utf-8"))


def get_current_weather(location: str) -> dict[str, Any]:
    geo = _get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": location, "count": 1, "language": "en", "format": "json"},
    )
    matches = geo.get("results") or []
    if not matches:
        return {"ok": False, "error": f"No location found for {location!r}."}
    place = matches[0]
    weather = _get_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
    )
    return {
        "ok": True,
        "location": {
            "name": place.get("name"),
            "region": place.get("admin1"),
            "country": place.get("country"),
        },
        "current": weather.get("current"),
        "units": weather.get("current_units"),
        "source": "Open-Meteo",
    }


@dataclass
class ToolResult:
    output: dict[str, Any]
    citations: list[dict[str, str]]


class ToolRegistry:
    def __init__(self, store: LocalVectorStore):
        self.store = store
        self.handlers: dict[str, Callable[..., ToolResult]] = {
            "search_knowledge_base": self.search_knowledge_base,
            "get_current_weather": self.weather,
            "evaluate_math": self.math_evaluate,
            "symbolic_math": self.math_symbolic,
            "analyze_numbers": self.math_statistics,
            "matrix_math": self.math_matrix,
            "convert_units": self.math_convert_units,
        }

    def search_knowledge_base(
        self, query: str, top_k: int, sources: list[str] | None = None
    ) -> ToolResult:
        results = self.store.search(query=query, top_k=top_k, sources=sources)
        citations = [
            {"citation": str(item["citation"]), "source": str(item["source"])}
            for item in results
        ]
        return ToolResult(output={"results": results}, citations=citations)

    def weather(self, location: str) -> ToolResult:
        return ToolResult(output=get_current_weather(location), citations=[])

    def math_evaluate(self, expression: str) -> ToolResult:
        return ToolResult(output=evaluate_math(expression), citations=[])

    def math_symbolic(
        self, operation: str, expression: str, variable: str
    ) -> ToolResult:
        return ToolResult(
            output=symbolic_math(operation, expression, variable), citations=[]
        )

    def math_statistics(self, values: list[float], operation: str) -> ToolResult:
        return ToolResult(output=analyze_numbers(values, operation), citations=[])

    def math_matrix(
        self,
        operation: str,
        matrix_a: list[list[float]],
        matrix_b: list[list[float]] | None,
    ) -> ToolResult:
        return ToolResult(
            output=matrix_math(operation, matrix_a, matrix_b), citations=[]
        )

    def math_convert_units(
        self, value: float, from_unit: str, to_unit: str
    ) -> ToolResult:
        return ToolResult(
            output=convert_units(value, from_unit, to_unit), citations=[]
        )

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        allowed_sources: list[str] | None = None,
    ) -> ToolResult:
        handler = self.handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")
        if name == "search_knowledge_base":
            return self.search_knowledge_base(**arguments, sources=allowed_sources)
        return handler(**arguments)
