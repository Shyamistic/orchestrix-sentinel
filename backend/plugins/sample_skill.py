# plugins/sample_skill.py
def run(inputs: dict) -> dict:
    """Example plugin skill - discovered dynamically at runtime"""
    title = inputs.get("title", "Untitled")
    return {
        "ticket_id": f"PLUG-{hash(title) % 10000}",
        "link": "https://plugins.orchestrix.ai/ticket/PLUG",
        "status": "CREATED"
    }
