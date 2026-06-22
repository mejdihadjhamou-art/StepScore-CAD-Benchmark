#!/usr/bin/env python3
import subprocess
import time
import urllib.request
import urllib.error
import socket
import sys

print("🔍 DIAGNOSTIC TEST\n" + "="*60)

# Test 1: Check Flask
print("\n1. Testing Flask installation...")
try:
    import flask
    print(f"   ✅ Flask version: {flask.__version__}")
except:
    print("   ❌ Flask not installed!")
    sys.exit(1)

# Test 2: Check if port 5000 is free
print("\n2. Checking port 5000...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('127.0.0.1', 5000))
sock.close()
if result == 0:
    print("   ⚠️  Port 5000 is already in use!")
    print("   Trying to kill existing process...")
    subprocess.run(["lsof", "-i", ":5000"], capture_output=False)
else:
    print("   ✅ Port 5000 is free")

# Test 3: Start server and test
print("\n3. Starting server...")
proc = subprocess.Popen(
    ["python", "3d_viewer_simple.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
time.sleep(3)

# Test 4: Check if server started
print("\n4. Testing server connection...")
try:
    response = urllib.request.urlopen('http://127.0.0.1:5000/', timeout=5)
    print(f"   ✅ Server responds with HTTP {response.status}")
    print(f"   ✅ Content-Type: {response.headers.get('Content-Type', 'unknown')}")
except urllib.error.HTTPError as e:
    print(f"   ❌ HTTP Error {e.code}: {e.reason}")
except Exception as e:
    print(f"   ❌ Connection failed: {e}")

# Test 5: Try alternate port
print("\n5. Trying alternate port 5001...")
proc.terminate()
time.sleep(1)

# Modify and start on different port
alt_server = """
from flask import Flask
from pathlib import Path
import pandas as pd

app = Flask(__name__)

EXCEL_PATH = Path(".stepscore_harness_runs/final73_anthropic_run_02/labeled_pairs_for_review.xlsx")
df = pd.read_excel(EXCEL_PATH)

@app.route('/')
def index():
    return '<h1>✅ Server Working!</h1><p>If you can see this, the server is fine.</p>'

if __name__ == '__main__':
    app.run(port=5001, host='0.0.0.0', debug=False)
"""

proc = subprocess.Popen(
    ["python", "-c", alt_server],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
time.sleep(3)

try:
    response = urllib.request.urlopen('http://127.0.0.1:5001/', timeout=5)
    print(f"   ✅ Port 5001 works! HTTP {response.status}")
except Exception as e:
    print(f"   ❌ Port 5001 also fails: {e}")

proc.terminate()

print("\n" + "="*60)
print("📋 DIAGNOSIS COMPLETE\n")
print("If port 5001 works but 5000 doesn't:")
print("   → Use: python 3d_viewer_alt.py (port 5001)")
print("   → Or: change default port in 3d_viewer_simple.py\n")
print("If both fail:")
print("   → Browser/firewall blocking localhost")
print("   → Try: Try a different browser or disable security extensions\n")
