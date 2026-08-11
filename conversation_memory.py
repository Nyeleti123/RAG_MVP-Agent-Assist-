"""
conversation_memory.py

Maintains the state of the conversation throughout the transcript.

Responsibilities
----------------
1. Parse transcript
2. Split into utterances
3. Segment conversation
4. Remove no-intent utterances
5. Store conversation memory
6. Store detected intents
7. Store generated checklists
"""

import re


class ConversationMemory:

    def __init__(self):

        self.reset()

        self.no_intent_keywords = [

            "hello",
            "hi",
            "good morning",
            "good afternoon",
            "good evening",
            "welcome",
            "thank you for calling",
            "please hold",
            "hold the line",
            "press",
            "option",
            "menu",
            "ivr",
            "voice mail",
            "voicemail",
            "call recorded",
            "quality assurance"

        ]

    # -------------------------------------------------------
    # RESET MEMORY
    # -------------------------------------------------------

    def reset(self):

        self.transcript = ""

        self.utterances = []

        self.segments = []

        self.meaningful_segments = []

        self.intent_history = []

        self.generated_checklists = []

        self.current_intent = None

        self.summary = {}

    # -------------------------------------------------------
    # LOAD TRANSCRIPT
    # -------------------------------------------------------

    def load_transcript(self, transcript):

        self.transcript = transcript

    # -------------------------------------------------------
    # SPLIT INTO UTTERANCES
    # -------------------------------------------------------

    def parse_utterances(self):

        self.utterances = []

        current_speaker = None

        for line in self.transcript.split("\n"):

            line = line.strip()

            if not line:
                continue

            if ":" in line:

                speaker, text = line.split(":", 1)
                speaker = speaker.strip()
                text = text.strip()

                current_speaker = speaker

                if text:
                    self.utterances.append({
                        "speaker": speaker,
                        "text": text
                    })

            else:

                if current_speaker is None:
                    continue

                if self.utterances and self.utterances[-1]["speaker"] == current_speaker:
                    self.utterances[-1]["text"] += " " + line
                else:
                    self.utterances.append({
                        "speaker": current_speaker,
                        "text": line
                    })

    # -------------------------------------------------------
    # SEGMENT CONVERSATION
    # -------------------------------------------------------

    def segment_conversation(self):

        self.segments = []

        for utterance in self.utterances:

            speaker = utterance["speaker"]

            text = utterance["text"]

            pieces = re.split(r"[.!?]+", text)

            for piece in pieces:

                piece = piece.strip()

                if len(piece) == 0:
                    continue

                self.segments.append({

                    "speaker": speaker,

                    "text": piece

                })

    # -------------------------------------------------------
    # REMOVE NO-INTENT SEGMENTS
    # -------------------------------------------------------

    def find_meaningful_segments(self):

        self.meaningful_segments = []

        for seg in self.segments:

            text = seg["text"].lower()

            if len(text.split()) < 4:
                continue

            ignore = False

            for keyword in self.no_intent_keywords:

                if re.search(r"\b" + re.escape(keyword) + r"\b", text):

                    ignore = True
                    break

            if ignore:
                continue

            self.meaningful_segments.append(seg)

    # -------------------------------------------------------
    # SAVE INTENT
    # -------------------------------------------------------
    
    def add_intent(self, intent, confidence):
        self.current_intent = intent
        self.intent_history.append({
            "intent": intent,

            "confidence": confidence,

            "status": "Pending"
            })
        
    # -------------------------------------------------------
    # SAVE GENERATED CHECKLIST
    # --------------------------------------------------------
    def add_checklist(
            self,
            intent,
            checklist,
            documents
    ):
        
        self.generated_checklists.append({
            
            "intent": intent,
            
            "checklist": checklist,
            
            "documents": documents,
            
            "status": "Pending"
            
        })

    # -------------------------------------------------------
    # BUILD SUMMARY
    # -------------------------------------------------------

    def build_summary(self):

        customer = sum(

            1 for u in self.utterances

            if u["speaker"].lower() == "customer"

        )

        agent = sum(

            1 for u in self.utterances

            if u["speaker"].lower() == "agent"

        )

        self.summary = {

            "turns": len(self.utterances),

            "customer_turns": customer,

            "agent_turns": agent,

            "segments": len(self.segments),

            "meaningful_segments": len(self.meaningful_segments),

            "intent_changes": len(self.intent_history),

            "generated_checklists": len(self.generated_checklists)

        }

    # -------------------------------------------------------
    # MAIN PROCESS FUNCTION
    # -------------------------------------------------------

    def process(self, transcript):

        self.reset()

        self.load_transcript(transcript)

        self.parse_utterances()

        self.segment_conversation()

        self.find_meaningful_segments()

        self.build_summary()

        return {

            "utterances": self.utterances,

            "segments": self.segments,

            "meaningful_segments": self.meaningful_segments,

            "intent_history": self.intent_history,

            "generated_checklists": self.generated_checklists,

            "current_intent": self.current_intent,

            "summary": self.summary

        }


if __name__ == "__main__":

    transcript = """
Customer: Hello
Agent: Good morning, how may I help you today?
Customer: I need to claim for my washing machine because it stopped working yesterday.
Agent: Certainly. May I have your policy number?
Customer: It is 123456.
"""

    memory = ConversationMemory()

    conversation = memory.process(transcript)

    print("=" * 70)
    print("MEANINGFUL SEGMENTS")
    print("=" * 70)

    for seg in conversation["meaningful_segments"]:
        print(seg)

    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(conversation["summary"])