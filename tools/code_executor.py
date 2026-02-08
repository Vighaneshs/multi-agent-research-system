"""Sandboxed Python code execution tool."""

from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import io
from contextlib import redirect_stdout, redirect_stderr


class CodeExecutorInput(BaseModel):
    code: str = Field(description="Python code to execute")
    timeout: int = Field(default=30, description="Maximum execution time in seconds")


@tool("execute_python", args_schema=CodeExecutorInput)
def execute_python_tool(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Run Python code in a restricted namespace and return stdout/stderr."""
    import math

    safe_namespace = {
        "__builtins__": {
            "print": print, "len": len, "range": range,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "sum": sum, "min": min, "max": max, "sorted": sorted,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "str": str, "int": int, "float": float, "bool": bool,
            "abs": abs, "round": round,
            "True": True, "False": False, "None": None,
        },
        "math": math,
    }

    # optionally expose pandas / numpy if installed
    for mod_name, alias in [("pandas", "pd"), ("numpy", "np")]:
        try:
            safe_namespace[alias] = __import__(mod_name)
        except ImportError:
            pass

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, safe_namespace)
        return {
            "success": True,
            "output": stdout_buf.getvalue() or "Executed successfully",
            "stderr": stderr_buf.getvalue(),
            "error": None,
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "output": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


@tool("analyze_data")
def analyze_data_tool(data: str, analysis_type: str = "summary") -> Dict:
    """Quick pandas analysis on JSON or CSV data. Returns summary, describe, or correlations."""
    try:
        import pandas as pd

        try:
            df = pd.read_json(io.StringIO(data))
        except Exception:
            df = pd.read_csv(io.StringIO(data))

        if analysis_type == "summary":
            return {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
            }
        elif analysis_type == "describe":
            return df.describe().to_dict()
        elif analysis_type == "correlations":
            numeric = df.select_dtypes(include=["number"]).columns
            if len(numeric) > 1:
                return df[numeric].corr().to_dict()
            return {"error": "Need at least 2 numeric columns for correlation"}
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}

    except ImportError:
        return {"error": "pandas not installed"}
    except Exception as e:
        return {"error": str(e)}
