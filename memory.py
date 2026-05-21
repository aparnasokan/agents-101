# memory.py
# Shared session state across all agents.
# Keeps conversation history so every agent has context of what's been said.

class SessionMemory:
    def __init__(self):
        self.history: list[dict] = []
        self.topics_covered: set[str] = set()
        self.agents_fired: int = 0
        self.message_count: int = 0

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})
        self.message_count += 1

    def add_assistant_message(self, content: str):
        if self.history and self.history[-1]["role"] == "assistant":
            self.history[-1]["content"] += f"\n\n{content}"
        else:
            self.history.append({"role": "assistant", "content": content})

    def add_topic(self, topic: str):
        if topic:
            self.topics_covered.add(topic.lower().strip())

    def record_agents_fired(self, count: int = 1):
        self.agents_fired += count

    def get_recent_history(self, turns: int = 6) -> list[dict]:
        """Return last N turns for agent context windows."""
        return self.history[-turns:]

    def get_full_history_text(self) -> str:
        """Flattened text of full session — used by Scribe."""
        lines = []
        for msg in self.history:
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            lines.append(f"{role}:\n{msg['content']}")
        return "\n\n---\n\n".join(lines)

    def get_stats(self) -> dict:
        return {
            "message_count": self.message_count,
            "agents_fired": self.agents_fired,
            "topics_covered": len(self.topics_covered),
            "topics_list": list(self.topics_covered),
        }

    def clear(self):
        self.__init__()


# Single shared instance used across the app
session = SessionMemory()
