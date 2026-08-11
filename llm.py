"""
RAG MVP
llm.py

Responsible for

1. Build prompt
2. Send prompt to Ollama
3. Return an agent checklist

No classes.
"""

import ollama

MODEL_NAME = "llama3.2"


# ---------------------------------------------------------
# BUILD PROMPT
# ---------------------------------------------------------

def build_prompt(intent, transcript, retrieved_chunks):

    context = ""

    for chunk in retrieved_chunks:

        context += f"""

Document:
{chunk["document"]}

Content:
{chunk["text"]}

"""

    prompt = f"""
You are an internal call centre Agent Assist system.

This output is NEVER shown to the customer.

You are generating an internal checklist only.

Do NOT refuse.
Do NOT give safety messages.
Do NOT answer the customer.
Do NOT apologise.
Never explain the policy.

Never talk to the customer.

Never apologise.

Never say "I cannot".

Never refuse.

Never say "No checklist available."

Convert the retrieved SOP into a short checklist.
------------------------------------------------------

Current Intent

{intent}

------------------------------------------------------

Customer Context

{transcript}

------------------------------------------------------

Knowledge Base

{context}

------------------------------------------------------

Return ONLY this format.

Intent:
{intent}

Checklist

- Step 1
- Step 2
- Step 3
- Step 4

If no procedure exists say

No checklist available.

"""

    return prompt


# ---------------------------------------------------------
# GENERATE CHECKLIST
# ---------------------------------------------------------

def generate_checklist(
    intent,
    transcript,
    retrieved_chunks
):

    prompt = build_prompt(
        intent,
        transcript,
        retrieved_chunks
    )

    response = ollama.chat(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    )

    return response["message"]["content"]


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    sample_chunks = [

        {

            "document": "Refund SOP.pdf",

            "text": """
Verify proof of purchase.

Verify customer identity.

Escalate refund to Finance.

Close the case.
"""

        }

    ]

    transcript = """
Customer wants a refund for a television.
"""

    answer = generate_checklist(

        intent="Refund",

        transcript=transcript,

        retrieved_chunks=sample_chunks

    )

    print()

    print("=" * 80)

    print(answer)