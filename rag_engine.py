"""
RAG MVP
rag_engine.py

Pipeline

Transcript
    ↓
Conversation Memory
    ↓
Intent Tracker
    ↓
Retriever
    ↓
LLM
    ↓
Agent Assist Output
"""

from conversation_memory import ConversationMemory
from intent_tracker import IntentTracker
from retriever import Retriever
from llm import generate_checklist


class RAGEngine:

    def __init__(self):

        print("Loading Conversation Memory...")
        self.memory = ConversationMemory()

        print("Loading Intent Tracker...")
        self.tracker = IntentTracker()

        print("Loading Retriever...")
        self.retriever = Retriever()

    # ----------------------------------------------------

    def process_transcript(self, transcript):

        # Step 1
        conversation = self.memory.process(transcript)

        # Only track intent on what the CUSTOMER says.
        # Support different transcript formats:
        # Customer:
        # Caller:
        # User
        #
        # Agent questions should never trigger intents.
    
        customer_segments = [
            
            seg
            for seg in conversation["meaningful_segments"]
            
            if seg.get("speaker", "").lower()
            in [
                "customer",
                "caller",
                "user"
                
            ]
            
        ]

        # Step 2
        detected_intents = self.tracker.track(customer_segments)

        results = []

        # Step 3
        for item in detected_intents[:3]:

            intent = item["intent"]
            confidence = item["confidence"]
            segment = item["segment"]

            # Write intent back into conversation memory
            self.memory.add_intent(intent, confidence)

            retrieved_chunks = self.retriever.retrieve(
                intent=intent,
                transcript=segment
            )

            checklist = generate_checklist(
                intent=intent,
                transcript=segment,
                retrieved_chunks=retrieved_chunks
            )

            # Write checklist back into conversation memory
            self.memory.add_checklist(
                intent,
                checklist,
                list({chunk["document"] for chunk in retrieved_chunks})
            )

            results.append({
                
                "intent": intent,
                
                "confidence": confidence,
                
                "segment": segment,
                
                "checklist": checklist,
                
                "documents": list(
                    {
                        chunk["document"]
                        for chunk in retrieved_chunks
                    }
                ),  # FIX: Added missing closing parenthesis
            
                "retrieved_chunks": retrieved_chunks,
            
                "status": "Pending"
            
            })  # FIX: Added missing closing parenthesis for results.append

        # Rebuild summary now that intents/checklists have been
        # recorded into memory (add_intent/add_checklist updated
        # self.memory.intent_history / generated_checklists after
        # build_summary() already ran inside self.memory.process()).
        self.memory.build_summary()

        return {

            "summary": self.memory.summary,

            "results": results

        }


# --------------------------------------------------------
# TEST
# --------------------------------------------------------

if __name__ == "__main__":

    transcript = """

Customer: Hello.

Agent: Good morning.

Customer:
My Samsung washing machine stopped working.

I would like a refund.

Agent:
Certainly.

Can I have your policy number?

Customer:
123456.

"""

    engine = RAGEngine()

    output = engine.process_transcript(
        transcript
    )

    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(output["summary"])

    print()

    print("=" * 80)
    print("INTENTS")
    print("=" * 80)

    for result in output["results"]:

        print()

        print(f"Intent      : {result['intent']}")
        print(f"Confidence  : {result['confidence']:.3f}")

        print()

        print("Checklist")

        print("---------------------------------------")

        print(result["checklist"])

        print()

        print("Documents")

        for doc in result["documents"]:
            print("-", doc)

        print()

        print("=" * 80)