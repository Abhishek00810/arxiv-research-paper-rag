import sys
sys.path.insert(0, 'src')

import json
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from agent import agent

load_dotenv()

# Override RAGAS default judge LLM (OpenAI) with Groq
judge_llm = LangchainLLMWrapper(ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct"))

# Override RAGAS default embeddings with local HuggingFace model
judge_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
)

# Load ground truth
with open("evals/ground_truth.json") as f:
    ground_truth_data = json.load(f)

questions = []
answers = []
contexts = []
ground_truths = []

total = len(ground_truth_data)
for i, item in enumerate(ground_truth_data):
    question = item["question"]
    print(f"[{i+1}/{total}] Running: {question[:60]}...")

    result = agent.invoke({
        "query": question,
        "original_query": "",
        "retrieved_chunks": [],
        "relevance_decision": "",
        "relevance_reason": "",
        "retry_count": 0,
        "final_answer": "",
        "sources": []
    })

    questions.append(question)
    answers.append(result["final_answer"])
    contexts.append([c["text"] for c in result["retrieved_chunks"]])  # actual chunk text
    ground_truths.append(item["ground_truth"])

    print(f"    retries: {result['retry_count']}")

# Build RAGAS dataset
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

print("\nRunning RAGAS evaluation...")
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_embeddings
)

print("\n=== RAGAS Scores ===")
print(results)

# Save results
scores = {
    "faithfulness": results["faithfulness"],
    "answer_relevancy": results["answer_relevancy"],
    "context_precision": results["context_precision"],
    "context_recall": results["context_recall"],
}
with open("evals/ragas_scores.json", "w") as f:
    json.dump(scores, f, indent=2)

print("\nSaved to evals/ragas_scores.json")
