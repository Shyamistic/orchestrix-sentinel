# sentinel_bridge.py
import requests
import time
from ibm_watsonx_orchestrate.agent_builder.tools import tool

# Configuration (Point this to your NGROK or Deployed Sentinel URL)
SENTINEL_API_URL = "https://your-sentinel-url.ngrok.io/api/v1"

@tool
def guardian_validate(user_intent: str) -> str:
    """
    Consults the Orchestrix Guardian Engine to check for policy violations.
    Returns 'APPROVED' or 'BLOCKED'.
    """
    try:
        # Call your Sentinel Backend
        response = requests.post(
            f"{SENTINEL_API_URL}/integrations/watsonx/guardrail", 
            json={"user_intent": user_intent}
        )
        data = response.json()
        
        if data["status"] == "BLOCKED":
            return f"BLOCKED: {data['message']}"
        return "APPROVED"
    except Exception as e:
        return f"ERROR: Sentinel unreachable - {str(e)}"

@tool
def reliable_execute(task_payload: str) -> str:
    """
    Executes a task with Orchestrix 'Self-Healing' wrapper. 
    If the task fails (simulated), Sentinel logs it to the Trust Ledger and retries.
    """
    # 1. Log Start to Sentinel Ledger
    requests.post(f"{SENTINEL_API_URL}/ledger/log", json={
        "event": "EXECUTION_START", "task": task_payload
    })

    # 2. Simulate Execution (with potential failure)
    try:
        # Simulate a 50% chance of "API Failure" to show off self-healing
        import random
        if random.random() < 0.5:
            raise Exception("503 Service Unavailable (Simulated)")
            
        result = f"Success: Processed {task_payload}"
        
        # 3. Log Success
        requests.post(f"{SENTINEL_API_URL}/ledger/log", json={
            "event": "EXECUTION_SUCCESS", "result": result
        })
        return result

    except Exception as e:
        # 4. THE SELF-HEALING MOMENT
        # Log Failure
        requests.post(f"{SENTINEL_API_URL}/ledger/log", json={
            "event": "FAILURE_DETECTED", "error": str(e)
        })
        
        # Wait (Simulate Healing)
        time.sleep(2)
        
        # Retry (Simulated Fix)
        requests.post(f"{SENTINEL_API_URL}/ledger/log", json={
            "event": "SELF_HEALING_TRIGGERED", "action": "Retry with Backoff"
        })
        return f"Success (Healed): Processed {task_payload} after recovery."