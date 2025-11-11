# socket_server.py (교체)
import socket, json
import time

class SocketServer:
    def __init__(self, host='127.0.0.1', port=12345, timeout=1, accept_timeout=10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.accept_timeout = accept_timeout
        self.buffer = ""
        self.last_action = None
        self._timeout_streak = 0   # ← 연속 타임아웃 카운터 초기화

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except Exception:
            pass
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(self.accept_timeout)
        print("[SERVER] Waiting for client...")
        self._accept_client()
        print("[SERVER] Client connected.")

    def _accept_client(self):
        while True:
            try:
                self.client_socket, _ = self.server_socket.accept()
                self.client_socket.settimeout(self.timeout)
                self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                except Exception:
                    pass
                self.buffer = ""
                self.last_action = None
                self._timeout_streak = 0   # ← 새 세션에서 타임아웃 누적 리셋
                self._in_reset = True 
                self._warmup_until = time.time() + 1.0
                return
            except socket.timeout:
                print("[SERVER] Still waiting for client...")
                continue

    def _reconnect(self):
        try:
            if hasattr(self, "client_socket"):
                try:
                    self.client_socket.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self.client_socket.close()
        except Exception:
            pass
        print("[SERVER] Re-accepting client...")
        self._accept_client()
        print("[SERVER] Reconnected.")

    def send(self, action_dict):
        try:
            # 1) dict만 허용. 문자열 "{...}"을 넘긴 실수를 보정
            if isinstance(action_dict, str):
                try:
                    action_dict = json.loads(action_dict)
                except json.JSONDecodeError:
                    print("[WARN] send() got string that is not JSON; wrapping as {'msg': ...}")
                    action_dict = {"msg": action_dict}

            # 2) 공백 제거한 JSON 문자열
            msg = json.dumps(action_dict, separators=(",", ":"))

            # 3) '@' 구분자 보장(중복 방지)
            if not msg.endswith("@"):
                msg = msg + "@"

            self.last_action = action_dict
            self.client_socket.sendall(msg.encode("utf-8"))

        except (socket.timeout, BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"[WARN] send() socket issue: {e} -> reconnect and retry")
            self._reconnect()
            self.client_socket.sendall(msg.encode("utf-8"))
        except Exception as e:
            print(f"[ERROR] send() unexpected: {e}")

    def keepalive(self):
        try:
            self.client_socket.sendall(b'{"KeepAlive":true}@')
        except (socket.timeout, BrokenPipeError, ConnectionResetError, OSError):
            self._reconnect()
            self.client_socket.sendall(b'{"KeepAlive":true}@')

    def receive(self):
        try:
            while True:
                try:
                    data = self.client_socket.recv(8192)
                except socket.timeout:
                    # (기존 로직) 타임아웃 누적/재접속 처리
                    print("[DEBUG] receive() timed out.")
                    if getattr(self, "_in_reset", False) or (getattr(self, "_warmup_until", 0) > time.time()):
                        return None
                    self._timeout_streak += 1
                    if self._timeout_streak >= 5:
                        print("[WARN] Timeout streak exceeded → reconnect to waiting UE")
                        self._reconnect()
                        self._timeout_streak = 0
                        self._in_reset = True
                        return "RECONNECT"
                    return None

                if not data:
                    print("[WARN] Empty read -> reconnect flow.")
                    self._reconnect()
                    continue

                decoded = data.decode("utf-8", errors="ignore")
                if decoded.strip() == "EpiDone":
                    print("[INFO] 'EpiDone' received directly from client.")
                    return "EpiDone"

                self.buffer += decoded

                last_json = None
                while "@" in self.buffer:
                    raw, self.buffer = self.buffer.split("@", 1)
                    raw = raw.strip()
                    if not raw or raw == '{"KeepAlive":true}':
                        continue

                    # 1차 시도
                    try:
                        last_json = json.loads(raw)
                        continue
                    except json.JSONDecodeError:
                        pass

                    # --- 간단 복구 1: 여분의 '{' 제거 (예: "{{...}" → "{...}") ---
                    fixed = raw.lstrip()
                    while fixed.startswith("{ {") or fixed.startswith("{{"):
                        fixed = fixed[1:].lstrip()

                    # --- 간단 복구 2: 닫힘 괄호 1개 보충 ---
                    if fixed.count("{") == fixed.count("}") + 1 and fixed.endswith("}")==False:
                        fixed = fixed + "}"

                    try:
                        last_json = json.loads(fixed)
                        continue
                    except json.JSONDecodeError:
                        # 이 조각은 건너뛰고 다음 조각 처리
                        print(f"[WARN] JSON parse failed (drop chunk): {raw[:120]}...")
                        continue

                if last_json is not None:
                    return last_json

        except OSError as e:
            print(f"[ERROR] receive() socket error: {e} -> reconnect and return None")
            self._reconnect()
            return None
        except Exception as e:
            print(f"[ERROR] receive() unexpected: {e}")
            return None

    def close(self):
        try:
            if hasattr(self, 'client_socket'):
                try:
                    self.client_socket.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self.client_socket.close()
        except Exception as e:
            print(f"[ERROR] client_socket close failed: {e}")
        try:
            if hasattr(self, 'server_socket'):
                self.server_socket.close()
        except Exception as e:
            print(f"[ERROR] server_socket close failed: {e}")
