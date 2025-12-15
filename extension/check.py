import zmq

HOST = "localhost"
PORT = 5555
ENDPOINT = f"tcp://{HOST}:{PORT}"

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 2000)   # 2s timeout
    sock.setsockopt(zmq.LINGER, 0)

    try:
        sock.connect(ENDPOINT)
        print(f"[OK] Connected to GR00T inference server at {ENDPOINT}")
        return 0
    except Exception as e:
        print(f"[ERROR] Could not connect to {ENDPOINT}: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
