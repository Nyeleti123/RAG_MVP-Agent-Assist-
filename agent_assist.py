"""
RAG MVP
agent_assist.py

Responsible for

1. Receive transcript
2. Run the complete RAG pipeline
3. Package results for the UI
4. Store approve/reject status
"""

from rag_engine import RAGEngine


class AgentAssist:

    def __init__(self):

        print("Loading Agent Assist...")

        self.engine = RAGEngine()  # FIX: Create an instance

        print("Agent Assist Ready.")

    # ----------------------------------------------------
    # RUN PIPELINE
    # ----------------------------------------------------

    def process(self, transcript):

        output = self.engine.process_transcript(transcript)  # FIX: Remove extra 'self'

        cards = []

        for result in output["results"]:

            cards.append({

                "intent": result["intent"],

                "confidence": round(result["confidence"], 3),

                "checklist": result["checklist"],

                "documents": result["documents"],

                "status": "Pending"

            })

        return {

            "summary": output["summary"],

            "current_intent":
                cards[-1]["intent"] if cards else None,

            "intent_history":
                [c["intent"] for c in cards],

            "cards": cards

        }

    # ----------------------------------------------------
    # APPROVE
    # ----------------------------------------------------

    def approve(self, card):

        card["status"] = "Approved"

        return card

    # ----------------------------------------------------
    # REJECT
    # ----------------------------------------------------

    def reject(self, card):

        card["status"] = "Rejected"

        return card


########################################################################
# TEST
########################################################################

if __name__ == "__main__":

    transcript = """

Customer: Hello.

Agent: Good morning.

Customer:
My Samsung washing machine stopped working.

Customer:
I want a refund.

Agent:
Can I have your policy number?

Customer:
123456

"""

    assist = AgentAssist()

    output = assist.process(transcript)

    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(output["summary"])

    print()

    print("=" * 80)
    print("DETECTED INTENTS")
    print("=" * 80)

    for intent in output["intent_history"]:

        print("✔", intent)

    print()

    print("=" * 80)
    print("CURRENT INTENT")
    print("=" * 80)

    print(output["current_intent"])

    print()

    print("=" * 80)
    print("AGENT CARDS")
    print("=" * 80)

    for card in output["cards"]:

        print()

        print("Intent:", card["intent"])

        print("Confidence:", card["confidence"])

        print()

        print("Checklist")

        print("---------------------")

        print(card["checklist"])

        print()

        print("Documents")

        for doc in card["documents"]:

            print("-", doc)

        print()

        print("Status:", card["status"])

        print()

        print("[Approve]   [Reject]")

        print("-" * 80)