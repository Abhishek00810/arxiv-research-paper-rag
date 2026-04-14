from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from retriever import retrieve
import json
import re
class AgentState(TypedDict):
    query: str
    original_query: str
    retrieved_chunks: List[dict]
    relevance_decision: str
    relevance_reason: str
    retry_count: int
    final_answer: str
    sources: list[str]

load_dotenv()


llm = ChatGroq(model="llama-3.3-70b-versatile")


def query_analyzer_node(state: AgentState) -> dict:
    query = state["query"]
    response = llm.invoke(f"""
      Classify this research question as simple or complex.                                                                                                             
      Simple: single source, factual                        
      Complex: requires multiple papers, comparison                                                                                                                     
      
    Question: {query}       

    Respond with JSON: {{"type": "simple" or "complex", "reason": "..."}}   
    """                                  
    )

    return {"original_query": query}


def retriever_node(state: AgentState) -> dict:
    query = state["query"]
    chunks = retrieve(query)
    return {"retrieved_chunks": chunks}

def grader_node(state: AgentState) -> dict:
    query = state["query"]
    chunks = state["retrieved_chunks"]

    chunks_text = "\n\n".join([c["text"] for c in chunks])

    response = llm.invoke(f"""
          You are evaluating retrieved context for a research question.                                                                                                   
                                                                                                                                                                        
      Question: {query}
                                                                                                                                                                        
      Retrieved chunks:                                                                                                                                                 
      {chunks_text}                                                                                                                                                   
                                                                                                                                                                        
      Can these chunks ALONE answer the question completely and accurately?                                                                                           
      Do not use any external knowledge. Only what is in these chunks.                                                                                                
                                                                                                                                                                        
      Respond with JSON: {{"decision": "sufficient" or "insufficient", "reason": "one sentence"}}    
    """
    )

    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)  
        result = json.loads(json_match.group())
        decision = result["decision"]
        reason = result["reason"]

    except:
        decision = "insufficient"
        reason = "failed to parse grader response"
    
    return {
        "relevance_decision": decision,
        "relevance_reason": reason
    }


def rewriter_node(state: AgentState) -> dict:
    original_query = state["original_query"]
    current_query = state["query"]
    reason = state["relevance_reason"]
    retry_count = state["retry_count"]
    response = llm.invoke(f"""
      A retrieval attempt failed to find relevant context.                                                                                                              
                                                          
      Original question: {original_query}                                                                                                                               
      Failed query: {current_query}                                                                                                                                   
      Why it failed: {reason}                                                                                                                                           
                                                                                                                                                                      
      Rewrite the query to target the missing information.
      Be more specific. Use different terminology.                                                                                                                      
      Return only the rewritten query, nothing else.
    """
    )

    return {
        "query": response.content.strip(),
        "retry_count": retry_count + 1
    }

def generator_node(state: AgentState) -> dict:
    query = state["query"]
    original_query = state["original_query"]
    chunks = state["retrieved_chunks"]

    chunks_text = "\n\n".join([                                                                                                                                       
          f"[{c['title']} | {c['section']}]\n{c['text']}"
          for c in chunks                                                                                                                                               
      ])              

    response = llm.invoke(f"""
      Answer the research question using ONLY the provided context.
      Do not use any knowledge outside these chunks.
      If context is insufficient, say so explicitly.

      Question: {original_query}

      Context:
      {chunks_text}

      Provide a clear, accurate answer.
      End with:
      Sources: [list each paper title and section used]
      """)     

    sources = [f"{c['title']} | {c['section']}" for c in chunks]   
    
    return {
        "final_answer": response.content,
        "sources": sources
    }
  

def route_after_grading(state: AgentState) -> str:
      if state["relevance_decision"] == "sufficient":                                                                                                                   
          return "generator"                                                                                                                                            
      elif state["retry_count"] >= 2:                                                                                                                                   
          return "generator"                                                                                                                                            
      else:                                                 
          return "rewriter"



graph = StateGraph(AgentState)

graph.add_node("query_analyzer", query_analyzer_node)
graph.add_node("retriever", retriever_node)
graph.add_node("grader", grader_node)
graph.add_node("rewriter", rewriter_node)                                                                                        
graph.add_node("generator", generator_node)

graph.set_entry_point("query_analyzer")
graph.add_edge("query_analyzer", "retriever")
graph.add_edge('retriever', "grader")
graph.add_conditional_edges("grader", route_after_grading)
graph.add_edge("rewriter", "retriever")
graph.add_edge("generator", END)

agent = graph.compile()

