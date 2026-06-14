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
    "previous_plan": [],
    "results": [],
    "answer": "",
    "approved": False,
    "reason": "",
    "retry_count": 0
}

state["question"] = input("Ask your question (or type exit)")


def planner(state):
    prompt = f"""

You are a planning agent    

Question:
{state["question"]}

Previous attempt feedback
(empty if first run):
Previous Plan:
{state["previous_plan"]}

Failure Reason:
{state["reason"]}

If feedback is provided create a plan that specifically addresses the issues mentioned.
Do not create the same plan if the feedback indicates missing information, incomplete reason.

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

    state["previous_plan"] = state["plan"]
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

        else:
            result = call_local_llm(step["action"])

        state["results"].append(
            {
                "action": action,
                "result": result
            }
        )

    return state


def answer(state):

    prompt = f"""
You are an analyst
    
Question:
{state["question"]}

Execution Results:
{state["results"]}

Analyze the results provided.
Answer the question.
Provide the answer in bullet points.

"""

    state["answer"] = call_local_llm(prompt)

    return state


def reflect(state):
    prompt = f"""

You are an Evaluating agent.
Question:
{state["question"]}

Plan:
{state["plan"]}

Results:
{state["results"]}

Answer:
{state["answer"]}

Determine whether the answer is supported by execution results.

Return JSON only:

Do not explain 
Do not use markdown.
Do not use ```json.

Follow the below format

{{
   "approved":true,
   "reason":"...."

}}

Follow the below constraints for "approved"

Approved is only:

true

or 

false
(if there is any chance of improvement)

"""
    result = call_cloud_llm(prompt)
    print("Reflect result: ", result)
    parsed = json.loads(result)
    state["approved"] = parsed["approved"]
    state["reason"] = parsed["reason"]

    print("Reason: ", state["reason"])

    return state


def reflection_router(state):

    print("Reflect: ", state["approved"])

    if state["approved"]:
        return "end"
    elif state["retry_count"] >= 1:
        return "end"
    else:
        state["retry_count"] += 1
        return "planner"


def end(state):
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

graph_builder.add_node(
    "reflect",
    reflect
)

graph_builder.add_node(
    "end",
    end
)

graph_builder.add_edge(
    "planner",
    "executor"
)

graph_builder.add_edge(
    "executor",
    "answer"
)

graph_builder.add_edge(
    "answer",
    "reflect"
)

graph_builder.add_conditional_edges(
    "reflect",
    reflection_router
)

graph_builder.set_entry_point(
    "planner"
)

graph_builder.set_finish_point(
    "end"
)

graph = graph_builder.compile()
graph.invoke(state)
