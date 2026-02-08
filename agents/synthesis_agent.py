"""Synthesis agent — pulls everything together into a final report."""

from typing import Dict, List
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent


class SynthesisAgent(BaseAgent):

    @property
    def system_prompt(self) -> str:
        return """You are a Research Synthesis Agent. Your job is to create 
comprehensive, well-structured research reports from gathered information.

Report Guidelines:
1. Start with a clear executive summary
2. Organize findings logically by theme or importance
3. Include specific data, statistics, and quotes when available
4. Properly attribute all sources
5. Highlight areas of consensus and disagreement
6. Note any gaps in the research
7. Provide actionable conclusions

Format the report in Markdown with clear sections:
- Use ## for main sections
- Use ### for subsections
- Use bullet points for lists
- Use > for important quotes
- Use tables for comparative data

Always maintain an objective, academic tone while being accessible."""

    def _format_search_results(self, results: List[Dict]) -> str:
        if not results:
            return "No search results available."

        chunks = []
        for i, r in enumerate(results, 1):
            chunks.append(
                f"Source {i}: {r.get('title', 'Untitled')}\n"
                f"URL: {r.get('source', 'Unknown')}\n"
                f"Content: {r.get('content', 'No content')}\n"
                f"Relevance: {r.get('relevance_score', 'N/A')}"
            )
        return "\n---\n".join(chunks)

    def _format_code_results(self, results: List[Dict]) -> str:
        if not results:
            return "No code analysis performed."

        chunks = []
        for i, r in enumerate(results, 1):
            status = "Success" if r.get("success") else "Failed"
            chunks.append(
                f"Analysis {i}: {status}\n"
                f"Code:\n```python\n{r.get('code', 'No code')}\n```\n"
                f"Output: {r.get('output', 'No output')}"
            )
        return "\n---\n".join(chunks)

    def run(self, query: str, search_results: List[Dict], code_results: List[Dict]) -> str:
        print(f"Synthesizing report for: {query[:50]}...")

        search_context = self._format_search_results(search_results)
        code_context = self._format_code_results(code_results)

        prompt = self.create_prompt_template(
            "Create a comprehensive research report for the following query:\n\n"
            "**Research Question:** {query}\n\n"
            "**Search Results:**\n{search_results}\n\n"
            "**Code Analysis Results:**\n{code_results}\n\n"
            "Generate a well-structured report that directly answers the research question, "
            "synthesizes information from all sources, includes relevant data and statistics, "
            "provides proper citations, highlights key insights and conclusions, "
            "and notes any limitations or areas needing further research.\n\n"
            "Format the report in Markdown."
        )

        chain = prompt | self.llm | StrOutputParser()
        report = chain.invoke({
            "query": query,
            "search_results": search_context,
            "code_results": code_context,
        })

        final_report = (
            f"# Research Report\n\n"
            f"**Query:** {query}\n\n"
            f"**Sources Analyzed:** {len(search_results)} search results, "
            f"{len(code_results)} code analyses\n\n---\n\n"
            f"{report}\n"
        )

        print(f"  Generated report ({len(final_report)} chars)")
        return final_report
