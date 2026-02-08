"""Planner agent — breaks research queries into subtasks."""

from typing import Dict, List
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from graph.state import ResearchTask


class PlannerOutput(BaseModel):
    reasoning: str = Field(description="Planning approach explanation")
    tasks: List[Dict] = Field(description="List of research tasks")


class PlannerAgent(BaseAgent):

    @property
    def system_prompt(self) -> str:
        return """You are a Research Planning Agent. Your job is to analyze research queries 
and break them down into a structured plan of subtasks.

For each task, you must specify:
1. task_id: A unique identifier (e.g., "task_1", "task_2")
2. description: What needs to be done
3. task_type: One of "search" (find information), "code" (run analysis), or "analysis" (reasoning)
4. dependencies: List of task_ids that must complete first (empty list if none)

Guidelines:
- Start with broad information gathering (search tasks)
- Follow with analysis tasks that depend on search results
- Use code tasks for data processing, calculations, or visualizations
- Keep tasks atomic and focused
- Identify clear dependencies to enable parallel execution where possible

Output your plan as valid JSON with the following structure:
{{
    "reasoning": "Your explanation of the planning approach",
    "tasks": [
        {{
            "task_id": "task_1",
            "description": "...",
            "task_type": "search|code|analysis",
            "dependencies": []
        }}
    ]
}}"""

    def run(self, query: str) -> Dict:
        prompt = self.create_prompt_template(
            "Create a research plan for the following query:\n\n{query}"
        )

        parser = JsonOutputParser(pydantic_object=PlannerOutput)
        chain = prompt | self.llm | parser
        result = chain.invoke({"query": query})

        tasks = []
        for t in result.get("tasks", []):
            tasks.append(ResearchTask(
                task_id=t["task_id"],
                description=t["description"],
                task_type=t["task_type"],
                dependencies=t.get("dependencies", []),
                status="pending",
            ))

        print(f"Planner created {len(tasks)} tasks:")
        for task in tasks:
            print(f"  [{task.task_type}] {task.description[:60]}")

        return {"reasoning": result.get("reasoning", ""), "tasks": tasks}
