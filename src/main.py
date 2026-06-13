# Planning agentic project where model creates a plan and then follows it through execution

from ollama import chat
from google import genai
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
import json
from tavily import TavilyClient

load_dotenv()
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


state = {
    "question": "",
    "plan": [],
    "results": [],
    "answer": ""
}

state["question"] = input("Ask your question (or type exit)")


def planner(state):
    prompt = f"""
Question:
{state["question"]}

Based on the question provided create a step by step plan
Provide the plan in JSON format
Do not explain.
Do not use markdown.
Do not use ```json.


Format:

{{
    "plan":[
        {{
            "action":"action description",
            "tool":"web_search"
        }}
    ]
}}

Available tools:

1. web_search
   - Current events
   - Recent information
   - Factual lookups

2. direct_llm
   - Summarization
   - Comparison
   - Reasoning
   - Explanation

Generate the plan for the question above.
"""

    parsed = json.loads(call_cloud_llm(prompt))
    state["plan"] = parsed["plan"]

    return state


def executor(state):

    for step in state["plan"]:

        tool = step["tool"]
        action = step["action"]

        print("Action: ", step["action"])
        print("Tool: ", step["tool"])

        if tool == "web_search":
            result = web_search(action)
            state["results"].append(
                {
                    "action": action,
                    "result": result
                }
            )

    return state


def answer(state):

    prompt = f"""

Question:
{state["question"]}

Execution Results:
{state["results"]}

Answer the question.
Provide the answer in bullet points.

"""

    state["answer"] = call_local_llm(prompt)

    print("Final Results: ", state["answer"])

    return state


def call_local_llm(prompt):
    response = chat(
        model="gemma4:e2b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return answer


def call_cloud_llm(prompt):
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text

    return answer


def web_search(action):
    results = tavily_client.search(
        query=action,
        max_results=5
    )
    context = ""
    for item in results["results"]:
        context += (
            f"{item["title"]}\n"
            f"{item["content"]}\n\n"
        )

    return context


graph_builder = StateGraph(dict)

graph_builder.add_node(
    "planner",
    planner
)

graph_builder.add_node(
    "executor",
    executor
)

graph_builder.add_node(
    "answer",
    answer
)

graph_builder.add_edge(
    "planner",
    "executor"
)

graph_builder.add_edge(
    "executor",
    "answer"
)

graph_builder.set_entry_point(
    "planner"
)

graph_builder.set_finish_point(
    "answer"
)

graph = graph_builder.compile()
graph.invoke(state)
