"""
RAG MVP - COMPLETE PIPELINE

Pipeline Flow

Question
    ↓
Intent Detection (Future)
    ↓
Retriever
    ↓
Top K Chunks
    ↓
LLM
    ↓
Grounded Answer

"""

import time
from retriever import retrieve
from llm import generate_answer


# OPTIONAL INTENT CLASSIFIER

def detect_intent(question):
    """
    Placeholder.

    Later this will call your Logistic Regression model.

    For now it simply returns Unknown.
    """

    return {

        "intent": "Unknown",

        "confidence": None

    }

# COMPLETE RAG PIPELINE

def answer_question(question):

    pipeline_start = time.time()

    # Step 1
    # Detect intent

    intent = detect_intent(question)

    # Step 2
    # Retrieve chunks

    retrieval = retrieve(question)

    # Step 3
    # Generate answer


    llm_response = generate_answer(retrieval)

    total_time = round(

        time.time() - pipeline_start,

        3

    )

    return {

        "question": question,

        "intent": intent,

        "retrieval": retrieval,

        "answer": llm_response,

        "pipeline_time": total_time

    }


# DISPLAY

def display_response(response):

    print()

    print("=" * 80)

    print("RAG PIPELINE")

    print("=" * 80)

    print()

    print("Question")

    print("------------------------------")

    print(response["question"])

    print()

    print("Intent")

    print("------------------------------")

    print(response["intent"]["intent"])

    print()

    print("Retrieved Documents")

    print("------------------------------")

    for doc in response["retrieval"]["documents"]:

        print(doc)

    print()

    print("Answer")

    print("------------------------------")

    print(response["answer"]["answer"])

    print()

    print("Pipeline Time")

    print("------------------------------")

    print(response["pipeline_time"], "seconds")

    print()


# MAIN

if __name__ == "__main__":

    print()

    print("=" * 80)

    print("RAG MVP")

    print("=" * 80)

    print()

    while True:

        question = input("Ask a question (type exit): ")

        if question.lower() == "exit":

            break

        print()

        print("Searching knowledge base...")

        response = answer_question(question)

        display_response(response)