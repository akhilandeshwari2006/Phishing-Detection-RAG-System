import time
import subprocess

print("Auto scanner running... Press Ctrl+C to stop.")

while True:
    print("\n[Scanner] Checking for new emails...")
    subprocess.run(["python", "gmail_fetcher.py"])
    subprocess.run(["python", "preprocess.py"])
    subprocess.run(["python", "detector.py"])
    subprocess.run(["python", "agent.py"])
    subprocess.run(["python", "rag.py"])
    subprocess.run(["python", "llm.py"])
    subprocess.run(["python", "output.py"])
    subprocess.run(["python", "notify.py"])
    print("[Scanner] Done. Waiting 5 minutes...")
    time.sleep(300)  # 300 seconds = 5 minutes