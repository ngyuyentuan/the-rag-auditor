import json, os, time
def log_event(path: str, event_type: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"ts": time.time(), "event_type": event_type, "payload": payload}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
