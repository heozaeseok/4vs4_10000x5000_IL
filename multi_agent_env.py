from socket_server import SocketServer
from ding.envs import BaseEnv
from ding.utils import ENV_REGISTRY
from ding.torch_utils.data_helper import to_ndarray
from gym import spaces
import numpy as np
import time

@ENV_REGISTRY.register('socket_multi_env')
class SocketMultiAgentEnv(BaseEnv):
    def __init__(self, cfg):
        self._cfg = cfg
        self.max_step = cfg.get('max_step', 1000)
        self.map_size = cfg.get('map_size', 4000)
        self.win_reward = cfg.get('win_reward', 100)

        self.num_agents = cfg.get('num_agents', 4)          # 아군 4명
        self.num_detectable = cfg.get('num_detectable', 4)  # 적 4명 감지

        # 팀1(아군) 4명, 팀0(적군) 4명
        self.agent_ids = ["AgentCharacter", "AgentCharacter2",
                        "AgentCharacter3", "AgentCharacter4"]  # 아군
        self.enemy_ids = ["AgentCharacter5", "AgentCharacter6",
                        "AgentCharacter7", "AgentCharacter8"]  # 적군


        self.last_data = {}
        self.soft_timeout_patience = cfg.get('soft_timeout_patience', 3)
        self.target_hz = cfg.get('target_hz', 60)
        self._next_deadline = None

        obs_dim = 4 + (self.num_agents - 1) * 4 + self.num_detectable * 4
        self._observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.valid_actions = list(range(0, 17))  # 0..7 이동, 8 근접사격, 9..16 지정사격
        self._action_space = spaces.Discrete(len(self.valid_actions))

        self.call_target_slots = ["AgentCharacter5", "AgentCharacter6", "AgentCharacter7", "AgentCharacter8"]
        self.call_target_map = {name: idx for idx, name in enumerate(self.call_target_slots)}

    def reset(self):
        print("=== [RESET CALL] ===")
        # 이전 서버가 살아있다면 안전 종료 (선택)
        try:
            if hasattr(self, "server"):
                self.server.close()
        except Exception:
            pass

        self.server = SocketServer()
        self.server.start()

        self.step_count = 0
        self._next_deadline = time.perf_counter()
        self.timeout_misses = 0
        self.last_done = {aid: False for aid in self.agent_ids}
        self.last_data = {}

        # 짧은 안정화
        time.sleep(0.1)
        # 잔여 버퍼 정리
        self.server.buffer = ""

        for attempt in range(50):
            data = self.server.receive()
            if isinstance(data, dict) and "Agents" in data:
                try:
                    obs_dict = {}

                    # 1) 넘어온 모든 에이전트 상태 캐시
                    for agent in data["Agents"]:
                        aid = agent["UnitID"]
                        self.last_data[aid] = agent
                        # 아군만 관측 생성
                        if aid in self.agent_ids:
                            obs_dict[aid] = to_ndarray(self._convert_obs(agent))

                    # 2) 아직 관측이 안 채워진 아군은 0으로 채움(길이 보장)
                    for aid in self.agent_ids:
                        if aid not in obs_dict:
                            obs_dict[aid] = to_ndarray(np.zeros(self._observation_space.shape, dtype=np.float32))

                    # 3) 초기 프레임에 적 상태가 비어 있으면 ‘살아있음’ 더미 주입
                    for eid in self.enemy_ids:
                        if eid not in self.last_data:
                            self.last_data[eid] = {
                                "UnitID": eid,
                                "HP": 100.0,  # alive
                                "LocX": 0.0, "LocY": 0.0, "LocZ": 0.0,
                                "DetectActors": []
                            }

                    # 4) 상태 반영 및 리셋 종료
                    self.current_obs = obs_dict
                    self._eval_episode_return = {aid: 0. for aid in self.agent_ids}

                    # ★ 탐지-접근 보상 상태 초기화 (정상 리셋에서도!)
                    self.prev_detect_dist = {aid: None for aid in self.agent_ids}
                    self.prev_detected    = {aid: False for aid in self.agent_ids}

                    setattr(self.server, "_in_reset", False)  # 리셋 종료: 서버 재전송 금지 해제
                    return obs_dict

                except Exception as e:
                    print(f"[RESET ERROR] Parsing failed: {e} (attempt {attempt+1})")

            time.sleep(0.1)

        # 모든 시도 실패 시 0 관측 반환
        print("[ERROR] No valid data received after 50 attempts.")
        obs_zeros = {aid: to_ndarray(np.zeros(self._observation_space.shape)) for aid in self.agent_ids}
        self.current_obs = obs_zeros
        # 탐지 기반 접근 보상 상태
        self.prev_detect_dist = {aid: None for aid in self.agent_ids}
        self.prev_detected    = {aid: False for aid in self.agent_ids}
        return obs_zeros

    def step(self, action: dict):
        assert isinstance(action, dict)

        # ✅ 송신 주기 고정(프레임 페이싱)
        if self._next_deadline is None:
            self._next_deadline = time.perf_counter()
        now = time.perf_counter()
        budget = 1.0 / float(self.target_hz)
        if now < self._next_deadline:
            time.sleep(self._next_deadline - now)
        self._next_deadline += budget

        self.step_count += 1

        if self.step_count >= self.max_step:
            print("[MAX_STEP] Reached max_step; ending episode.")
            # UE에도 리셋 트리거 전송: Counts + ResetMap
            self.server.send({
                "Agents": [
                    {"UnitID": aid, "Order": 0, "Pitch": 0, "Yaw": 0, "Roll": 0}
                    for aid in self.agent_ids
                ],
                "Counts": 1500
            })
            self.server.send({"ResetMap": True})
            setattr(self.server, "_in_reset", True)   # 다음 reset까지 재전송 금지
            self.server.last_action = None
            return self._dummy_timestep(done_all=True)

        # 아군 액션(죽었으면 0) + 적군 룰베이스 액션을 묶어서 전송
        def _zero_rot():
            # UE 파서 에러 방지용 기본 회전값
            return {"Pitch": 0.0, "Yaw": 0.0, "Roll": 0.0}

        ''' 모든행동 포함
        ally_actions = [
            {
                "UnitID": aid,
                "Order": 0 if self.last_done.get(aid, False) else int(action[aid].item()),
                **_zero_rot(),  # ← 추가
            } for aid in self.agent_ids
        ]
        '''

        #8번행동 제거(PPO모델 인덱싱 오류 해결)
        ally_actions = []
        for aid in self.agent_ids:
            if self.last_done.get(aid, False):
                order = 0
            else:
                act_idx = int(action[aid].item())      # PPO가 뽑은 0~9 인덱스
                order   = self.valid_actions[act_idx]  # UE에 보낼 실제 액션 번호 (8 제외)
            ally_actions.append({
                "UnitID": aid,
                "Order": order,
                **_zero_rot(),
            })



        # 적군행동 선택
        enemy_actions = self._enemy_seek_or_shoot_actions()
        #enemy_actions = self._friendly_initial_actions()

        try:
            self.server.send({"Agents": ally_actions + enemy_actions})
        except (BrokenPipeError, ConnectionResetError, OSError):
            return self._dummy_timestep(done_all=False)

        data = self.server.receive()

        # 명시적 종료 신호는 즉시 종료
        if isinstance(data, str) and data.strip() == "EpiDone":
            setattr(self.server, "_in_reset", True)
            self.server.last_action = None
            print("[INFO] EpiDone received. Forcing episode termination.")
            return self._dummy_timestep(done_all=True)

        # 소켓 서버가 재접속을 수행했음을 알림 → 이번 에피소드 종료 후 reset()로 재연결
        if data == "RECONNECT":
            print("[INFO] Server requested reconnect → ending episode for reset()")
            setattr(self.server, "_in_reset", True)
            self.server.last_action = None
            return self._dummy_timestep(done_all=True)

        if data is None:
            self.timeout_misses += 1
            if self.timeout_misses >= getattr(self, "soft_timeout_patience", 3):
                print("[WARN] Soft timeouts exceeded; end episode to allow reset()")
                self.timeout_misses = 0
                setattr(self.server, "_in_reset", True)
                self.server.last_action = None
                return self._dummy_timestep(done_all=True)
            # 임계 전에는 프레임만 유지
            return self._dummy_timestep(done_all=False)

        if not isinstance(data, dict) or "Agents" not in data:
            print("[ERROR] No valid agent data received. End episode for reset().")
            setattr(self.server, "_in_reset", True)
            self.server.last_action = None
            self.timeout_misses += 1
            return self._dummy_timestep(done_all=True)

        self.timeout_misses = 0

        if not self._has_all_allies(data):
            print("[FRAME] missing ally state; skipping frame.")
            return self._dummy_timestep(done_all=False)

        # --- obs/rew/done/info 기본 생성 ---
        obs, rew, done, info = {}, {}, {}, {}
        for agent in data["Agents"]:
            aid = agent.get("UnitID")
            self.last_data[aid] = agent

            if aid in self.agent_ids:
                obs[aid]  = to_ndarray(self._convert_obs(agent))
                ue_reward = float(agent.get("Reward", 0.0))
                rew[aid]  = ue_reward
                done[aid] = bool(agent.get("isDone", False)) or (float(agent.get("HP", 0)) <= 0)
                info[aid] = {}
                # ★ 추가: 내부 done 캐시 갱신
                self.last_done[aid] = done[aid]

        # 최신 관측 저장
        self.current_obs = obs

        # ==== 탐지-접근: 감지된 스텝에만 ±0.05, 아니면 0 ====
        APPROACH_REWARD = 0.5
        EPS = 1e-4

        # obs 레이아웃: [자기(4)] + [아군(4)*(len(self.agent_ids)-1)] + [적 슬롯(4)*num_detectable]
        start_enemy_base = 4 + (len(self.agent_ids) - 1) * 4

        for aid in self.agent_ids:
            if self.last_done.get(aid, True) or aid not in obs:
                # 종료/사망/누락 시 추적 해제
                self.prev_detect_dist[aid] = None
                self.prev_detected[aid]    = False
                continue

            ov = obs[aid]  # 1D np.ndarray
            sx, sy = float(ov[0]), float(ov[1])

            # 이번 스텝에 실제 감지된 적들 중 최근거리
            nearest = None
            for idx in range(self.num_detectable):
                base = start_enemy_base + idx * 4
                ex, ey, _, flag = float(ov[base]), float(ov[base + 1]), float(ov[base + 2]), float(ov[base + 3])
                if flag >= 0.5:
                    dx, dy = sx - ex, sy - ey
                    d = (dx*dx + dy*dy) ** 0.5
                    nearest = d if (nearest is None or d < nearest) else nearest

            detected_now = (nearest is not None)

            # 감지된 스텝에서만 진행 비교, 아니면 보상 0
            if detected_now and self.prev_detected.get(aid, False) and (self.prev_detect_dist.get(aid) is not None):
                prev_d = self.prev_detect_dist[aid]
                if nearest < prev_d - EPS:
                    rew[aid] += APPROACH_REWARD      # 가까워짐 → +0.05
                elif nearest > prev_d + EPS:
                    rew[aid] -= APPROACH_REWARD      # 멀어짐 → -0.05
                else:
                    rew[aid] += 0.0                  # 거의 변화 없음
            else:
                rew[aid] += 0.0                      # 감지 안 됨 → 0

            # 상태 업데이트
            self.prev_detect_dist[aid] = (nearest if detected_now else None)
            self.prev_detected[aid]    = detected_now

        # --- 방어: 누락된 에이전트에 기본값 보장 ---
        for aid in self.agent_ids:
            if aid not in obs:
                obs[aid] = to_ndarray(np.full(self._observation_space.shape, -1.0, dtype=np.float32))
            if aid not in rew:
                rew[aid] = 0.0
            if aid not in done:
                done[aid] = False
            if aid not in info:
                info[aid] = {}

        # --- 에피소드 종료 여부 판단 ---
        done["__all__"] = all(done[aid] for aid in self.agent_ids)

        if done["__all__"]:
            # 1) max_step/타임아웃으로 강제 종료된 경우에는 보너스/패널티 없음
            if self.step_count >= self.max_step:
                for aid in self.agent_ids:
                    rew[aid] -= self.win_reward
                print("[ENV] Max step reached: allies treated as LOSS → loss penalty to all.")
            else:
                # 2) 실제 종료: 현재 프레임의 HP로 '아군만' 승패 판정
                alive_any_ally = False
                if isinstance(data, dict) and "Agents" in data:
                    for agent in data["Agents"]:
                        if agent.get("UnitID") in self.agent_ids and float(agent.get("HP", 0)) > 0:
                            alive_any_ally = True
                            break

                if alive_any_ally:
                    for aid in self.agent_ids:
                        rew[aid] += self.win_reward
                    print("[ENV] Team win: at least one ALLY alive → all get win_reward.")
                else:
                    for aid in self.agent_ids:
                        rew[aid] -= self.win_reward
                    print("[ENV] Team loss: allies all dead → loss penalty to all.")
        else:
            # 진행 중 추가 보상 항목이 있다면 여기서 더하세요.
            pass

        return {
            'obs': obs,
            'reward': rew,
            'done': done,
            'info': info
        }

    def _dummy_timestep(self, done_all: bool = False):
        dummy_obs = {aid: to_ndarray(self.current_obs.get(aid, np.zeros(self._observation_space.shape)))
                     for aid in self.agent_ids}
        rew = {aid: 0.0 for aid in self.agent_ids}
        done = {aid: done_all for aid in self.agent_ids}
        done["__all__"] = done_all
        info = {aid: {} for aid in self.agent_ids}
        return {
            'obs': dummy_obs,
            'reward': rew,
            'done': done,
            'info': info
        }

    def _convert_obs(
        self,
        data,
        center=(5780.0, 6110.0, 100.0),
        detect_range: float = 20000.0,
    ):
        """
        Unreal 파라미터를 직접 받는 버전:
        - center: (cx, cy, cz) 맵 중심 좌표
        - detect_range: 모든 객체의 공통 감지 사거리 (m 단위)
        """
        cx, cy, cz = center
        self_id = data.get("UnitID", "")
        sx = float(data.get("LocX", 0))
        sy = float(data.get("LocY", 0))
        sz = float(data.get("LocZ", 0))
        self_hp = float(data.get("HP", 0)) / 100.0

        # === 1️⃣ 자기 좌표 (절대좌표) ===
        abs_x = (sx - cx) / self.map_size
        abs_y = (sy - cy) / self.map_size
        abs_z = (sz - cz) / self.map_size

        obs = [abs_x, abs_y, abs_z, self_hp]

        # === 2️⃣ 아군 정보 (가까운 순 정렬) ===
        ally_rel = []
        for other in self.agent_ids:
            if other == self_id:
                continue
            od = self.last_data.get(other)
            if od:
                ox, oy, oz = float(od.get("LocX", 0)), float(od.get("LocY", 0)), float(od.get("LocZ", 0))
                ohp = float(od.get("HP", 0)) / 100.0
                dx, dy, dz = ox - sx, oy - sy, oz - sz
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                ally_rel.append((dist, dx, dy, dz, ohp))
            else:
                ally_rel.append((float("inf"), 0.0, 0.0, 0.0, 0.0))

        ally_rel.sort(key=lambda x: x[0])
        ally_rel = ally_rel[: (self.num_agents - 1)]

        for _, dx, dy, dz, ohp in ally_rel:
            obs.extend([
                np.clip(dx / detect_range, -1, 1),
                np.clip(dy / detect_range, -1, 1),
                np.clip(dz / detect_range, -1, 1),
                ohp,
            ])

        # === 3️⃣ 적군 정보 (DetectActors 기준, 감지 반경 정규화) ===
        obs.extend([0.0, 0.0, 0.0, 0.0] * self.num_detectable)
        detected = data.get("DetectActors", [])
        enemy_rel = []

        for det in detected:
            name = det.get("Name", "")
            if name in self.enemy_ids:
                ex, ey, ez = float(det.get("LocX", 0)), float(det.get("LocY", 0)), float(det.get("LocZ", 0))
                dx, dy, dz = ex - sx, ey - sy, ez - sz
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                enemy_rel.append((dist, name, dx, dy, dz))

        enemy_rel.sort(key=lambda x: x[0])
        for idx, (_, name, dx, dy, dz) in enumerate(enemy_rel[: self.num_detectable]):
            base = 4 + (self.num_agents - 1) * 4 + idx * 4
            obs[base: base + 4] = [
                np.clip(dx / detect_range, -1, 1),
                np.clip(dy / detect_range, -1, 1),
                np.clip(dz / detect_range, -1, 1),
                1.0,  # 감지됨 플래그
            ]

        arr = np.round(np.array(obs, dtype=np.float32), 3)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr


    def _friendly_initial_actions(self, attack_range: float = 1500.0, tol: float = 0.0):
        """
        아군 초기(룰-베이스) 행동 생성:
        - 관측상 적이 없으면: 1~8 중 랜덤
        - 관측상 적이 있고, 사거리 밖이면: 가장 가까운 적에게 접근(1~4/5~8)
        - 관측상 적이 있고, 사거리 안이면: 사격(9)

        Unreal DoAction 매핑(예시; 프로젝트 매핑에 맞춰 조정 가능):
        1: +X(앞)   2: -X(뒤)   3: -Y(좌)   4: +Y(우)
        5: +X+Y(앞우 대각)   6: +X-Y(앞좌 대각)
        7: -X+Y(뒤우 대각)   8: -X-Y(뒤좌 대각)
        9: 사격

        인자:
        attack_range: 사거리(단위: 맵 좌표 단위)
        tol         : 목표에 매우 가까울 때 무작위 이동 허용 임계(0이면 꺼짐)
        필요 필드:
        self.agent_ids  : 아군 유닛 ID 리스트
        self.enemy_ids  : 적군 유닛 ID 리스트
        self.last_data  : 각 유닛 상태 dict(LocX, LocY, LocZ, HP, isDone, DetectActors 등)
        """
        import numpy as np
        rng = np.random

        def nearest_enemy(my_x, my_y, my_z, detections):
            """
            detections: 감지된 적 배우 리스트 (ed['DetectActors']와 동일 구조)
                        각 원소는 { 'Name': <enemy_id>, 'LocX':..., 'LocY':..., 'LocZ':... }
            반환: (tx,ty,tz, dist, det_dict)  혹은 (None,... ) if empty
            """
            best = None
            best_d2 = None
            for d in detections:
                if d.get("Name") not in self.enemy_ids:
                    continue
                ex = float(d.get("LocX", 0.0))
                ey = float(d.get("LocY", 0.0))
                ez = float(d.get("LocZ", 0.0))
                dx, dy, dz = (ex - my_x), (ey - my_y), (ez - my_z)
                d2 = dx*dx + dy*dy + dz*dz
                if (best_d2 is None) or (d2 < best_d2):
                    best_d2 = d2
                    best = (ex, ey, ez, np.sqrt(d2), d)
            if best is None:
                return None, None, None, None, None
            return best

        def choose_move(dx: float, dy: float, diag_ratio: float = 0.4) -> int:
            ax, ay = abs(dx), abs(dy)
            if ax <= tol and ay <= tol:
                return int(rng.choice(range(0, 8)))  # 0..7
            if ax > 0 and ay > 0 and min(ax, ay) / max(ax, ay) >= diag_ratio:
                if dx >= 0 and dy >= 0: return 4
                if dx >= 0 and dy <  0: return 6  # +X -Y
                if dx <  0 and dy >= 0: return 5  # -X +Y
                return 7                           # -X -Y
            if ax >= ay: return 0 if dx > 0 else 1
            else:        return 3 if dy > 0 else 2

        def fixed_shoot_action(target_name: str) -> int:
            """Call Target Array에 있으면 9+슬롯, 없으면 8(근접사격)"""
            idx = None
            if hasattr(self, "call_target_map"):
                idx = self.call_target_map.get(target_name, None)
            if idx is None or not (0 <= idx <= 7):
                return 8
            return 9 + int(idx)  # 9..16

        actions = []
        r2 = attack_range * attack_range

        for aid in self.agent_ids:
            ad = self.last_data.get(aid, {})
            hp = ad.get("HP", None)
            is_dead = bool(ad.get("isDone", False)) or (hp is not None and float(hp) <= 0)
            if is_dead:
                actions.append({"UnitID": aid, "Order": 0, "Pitch": 0, "Yaw": 0, "Roll": 0})
                continue

            sx = float(ad.get("LocX", 0.0)); sy = float(ad.get("LocY", 0.0)); sz = float(ad.get("LocZ", 0.0))
            dets = [d for d in ad.get("DetectActors", []) if d.get("Name") in self.enemy_ids]

            if len(dets) == 0:
                order = int(np.random.randint(0, 8))  # 적 미발견: 0..7 랜덤 이동
            else:
                # 가장 가까운 적(3D 거리)
                best = None; best_d2 = None
                for d in dets:
                    ex = float(d.get("LocX", 0.0)); ey = float(d.get("LocY", 0.0)); ez = float(d.get("LocZ", 0.0))
                    dx, dy, dz = (ex - sx), (ey - sy), (ez - sz)
                    d2 = dx*dx + dy*dy + dz*dz
                    if best_d2 is None or d2 < best_d2:
                        best_d2, best = d2, d
                if best is None:
                    order = int(np.random.randint(0, 8))
                else:
                    if best_d2 <= r2:
                        # 사거리 안: 지정사격(가능하면 9..16), 아니면 8
                        order = fixed_shoot_action(best.get("Name", ""))
                    else:
                        # 사거리 밖: 접근
                        tx, ty = float(best.get("LocX", 0.0)), float(best.get("LocY", 0.0))
                        order = choose_move(tx - sx, ty - sy)

            actions.append({"UnitID": aid, "Order": int(order), "Pitch": 0, "Yaw": 0, "Roll": 0})

        return actions



    def _enemy_seek_or_shoot_actions(self, x_target=2070.0, y_target=6140.0, tol=0.0):
        """
        행동 매핑 (언리얼 DoAction과 일치):
        0: +X (앞으로 이동)
        1: -X (뒤로 이동)
        2: -Y (왼쪽 이동)
        3: +Y (오른쪽 이동)
        8: 사격
        """
        def axis_move(dx: float, dy: float) -> int:
            if abs(dx) <= tol and abs(dy) <= tol:
                return np.random.choice([0, 1, 2, 3])  # 가까우면 임의 이동
            if abs(dx) >= abs(dy):
                return 0 if dx > 0 else 1  # +X → 0, -X → 1
            else:
                return 3 if dy > 0 else 2  # +Y → 3, -Y → 2

        actions = []
        for eid in self.enemy_ids:
            ed = self.last_data.get(eid, {})
            hp = ed.get("HP", None)
            is_dead = bool(ed.get("isDone", False)) or (hp is not None and float(hp) <= 0)
            if is_dead:
                actions.append({"UnitID": eid, "Order": 0, "Pitch": 0, "Yaw": 0, "Roll": 0})
                continue

            detections = [det for det in ed.get("DetectActors", []) if det.get("Name") in self.agent_ids]
            sx, sy = float(ed.get("LocX", 0)), float(ed.get("LocY", 0))

            if len(detections) == 0:
                # 감지 X: 70% 지정좌표 이동 / 30% 랜덤
                if np.random.rand() < 0.7:
                    dx, dy = (x_target - sx), (y_target - sy)
                    order = axis_move(dx, dy)
                else:
                    order = np.random.choice([0, 1, 2, 3, 8])
            else:
                # 감지 O: 가장 가까운 적 추적
                closest_det = min(
                    detections,
                    key=lambda d: (float(d.get("LocX", 0)) - sx) ** 2 + (float(d.get("LocY", 0)) - sy) ** 2
                )
                tx, ty = float(closest_det.get("LocX", 0)), float(closest_det.get("LocY", 0))

                if np.random.rand() < 0.7:
                    dx, dy = (tx - sx), (ty - sy)
                    order = axis_move(dx, dy)
                else:
                    order = 8  # 사격

            actions.append({
                "UnitID": eid,
                "Order": int(order),
                "Pitch": 0, "Yaw": 0, "Roll": 0,
            })
        return actions



    def _has_all_allies(self, data):
        """
        Allies(아군) 데이터가 모두 수신되었는지 확인.
        """
        if not isinstance(data, dict) or "Agents" not in data:
            return False
        seen = {a.get("UnitID") for a in data["Agents"]}
        return all(aid in seen for aid in self.agent_ids)
    
    def close(self):
        self.server.close()

    def start(self):
        self.server.start()

    def seed(self, seed: int = 0, dynamic_seed: bool = False) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def random_action(self) -> dict:
        return {aid: np.array([self._action_space.sample()], dtype=np.int64) for aid in self.agent_ids}

    def __repr__(self):
        return "SocketMultiAgentEnv()"

    @property
    def observation_space(self):
        return {aid: self._observation_space for aid in self.agent_ids}

    @property
    def action_space(self):
        return {aid: self._action_space for aid in self.agent_ids}

    @property
    def reward_space(self):
        return {aid: spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32) for aid in self.agent_ids}
