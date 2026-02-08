"""Code agent — generates and runs Python for data analysis."""

from typing import Dict, List, Optional, Any
import io
import math
import traceback
from contextlib import redirect_stdout, redirect_stderr

from agents.base_agent import BaseAgent
from graph.state import CodeExecutionResult


class CodeAgent(BaseAgent):

    @property
    def system_prompt(self) -> str:
        return """You are a Code Analysis Agent. Your job is to write and execute 
Python code to analyze data, perform calculations, or create visualizations.

Guidelines:
1. Write clean, well-documented Python code
2. Use standard libraries (pandas, numpy, matplotlib) when appropriate
3. Always include error handling
4. Print results clearly with labels
5. For visualizations, save to file rather than showing interactively

When given context (like search results), incorporate relevant data into your analysis.

Output your code in a code block like this:
```python
# Your code here
```

After the code, explain what it does and what results to expect."""

    def _extract_code(self, response: str) -> Optional[str]:
        """Pull the first python (or bare) fenced code block out of an LLM response."""
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + len("```")
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        return None

    def _create_safe_namespace(self) -> Dict[str, Any]:
        """Restricted globals dict for exec(). Keeps things reasonably sandboxed."""
        safe_builtins = {
            'print': print, 'len': len, 'range': range,
            'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
            'sum': sum, 'min': min, 'max': max, 'sorted': sorted,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'str': str, 'int': int, 'float': float, 'bool': bool,
            'abs': abs, 'round': round,
        }

        namespace = {'__builtins__': safe_builtins, 'math': math}

        # optionally expose pandas / numpy if installed
        try:
            import pandas as pd
            namespace['pd'] = pd
        except ImportError:
            pass
        try:
            import numpy as np
            namespace['np'] = np
        except ImportError:
            pass

        return namespace

    def _execute_code(self, code: str, timeout: int = 30) -> CodeExecutionResult:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        namespace = self._create_safe_namespace()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, namespace)

            output = stdout_buf.getvalue()
            return CodeExecutionResult(
                code=code,
                output=output or "Code executed successfully (no output)",
                success=True,
                error=stderr_buf.getvalue() or None,
            )
        except Exception as e:
            return CodeExecutionResult(
                code=code,
                output=stdout_buf.getvalue(),
                success=False,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )

    def _format_context(self, context: List[Dict]) -> str:
        if not context:
            return "No additional context provided."
        lines = ["Available data from search results:\n"]
        for i, item in enumerate(context[:5], 1):
            lines.append(f"{i}. {item.get('title', 'Untitled')}")
            lines.append(f"   {item.get('content', '')[:200]}...\n")
        return "\n".join(lines)

    def run(self, task_description: str, context: List[Dict] = None) -> Dict:
        print(f"Generating code for: {task_description[:60]}...")

        context_str = self._format_context(context or [])
        prompt = self.create_prompt_template(
            "Task: {task}\n\nContext:\n{context}\n\n"
            "Write Python code to accomplish this task. Use the context data if relevant."
        )

        chain = prompt | self.llm
        response = chain.invoke({"task": task_description, "context": context_str})
        code = self._extract_code(response.content)

        if not code:
            return {
                "code": "",
                "output": "Failed to generate valid code",
                "success": False,
                "error": "No code block found in LLM response",
            }

        print(f"  Generated {len(code.splitlines())} lines of code")
        result = self._execute_code(code)

        status = "ok" if result.success else f"failed — {result.error[:50]}"
        print(f"  Execution: {status}")

        return {
            "code": result.code,
            "output": result.output,
            "success": result.success,
            "error": result.error,
        }
