import torch.nn.functional as F
import numpy as np
import torch
from easydict import EasyDict
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.utils import set_pkg_seed
from ding.torch_utils import to_device
from tensorboardX import SummaryWriter
from multi_agent_env import SocketMultiAgentEnv
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime
import threading, time

class EnvKeepAlive:
    def __init__(self, env, interval=0.3):
        self.env = env
        self.interval = interval
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self._stop:
            try:
                if hasattr(self.env, "server") and hasattr(self.env.server, "keepalive"):
                    self.env.server.keepalive()
            except Exception:
                pass
            time.sleep(self.interval)
    def __enter__(self):
        self._t.start(); return self
    def __exit__(self, exc_type, exc, tb):
        self._stop = True; self._t.join(timeout=1)


# ===== 설정 =====
TOTAL_STEPS = 10000000
reward_scale = 0.1
win_reward = 100

PLOT_INTERVAL = 10000000
next_plot_step = PLOT_INTERVAL

# ===== 환경 초기화 =====
env = SocketMultiAgentEnv({
    'map_size': 5000,
    'max_step': 1000,
    'win_reward': win_reward,
    'num_detectable': 4,  # 적 4명 감지
    'num_agents': 4       # 아군 4명
})

agent_ids = env.agent_ids
OBS_SHAPE = env.observation_space[agent_ids[0]].shape[0]
ACTION_DIM = env.action_space[agent_ids[0]].n

config = dict(
    type='ppo',
    action_space='discrete',
    cuda=True,
    multi_gpu=False,
    on_policy=True,
    priority=False,
    priority_IS_weight=False,
    multi_agent=True,
    recompute_adv=True,
    transition_with_policy_data=True,
    nstep_return=False,

    model=dict(
        obs_shape=OBS_SHAPE,
        action_shape=ACTION_DIM,
        encoder_hidden_size_list=[256, 256, 128],
        actor_head_hidden_size=128,
        actor_head_layer_num=3,
        critic_head_hidden_size=256,
        critic_head_layer_num=3,
        share_encoder=False,  # actor/critic 분리 권장
        action_space='discrete',
        activation=torch.nn.SiLU(),  # ReLU보다 부드러운 비선형성
        norm_type='LN',
    ),
    learn=dict(
        epoch_per_collect=4,       # 한 번 데이터 수집 후(collect) 학습 에폭 반복 횟수
                                # ↑ 높이면 같은 데이터로 더 많이 업데이트 → 데이터 효율 ↑, 과적합 위험 ↑
                                # ↓ 낮추면 업데이트 보수적, 수렴 느려짐

        batch_size=512,             # 학습 시 미니배치 크기
                                # ↑ 크면 gradient 안정성 ↑, GPU 효율 ↑, 하지만 업데이트 빈도 ↓
                                # ↓ 작으면 업데이트 빈도 ↑, gradient 변동성 ↑

        learning_rate=3e-4,        # 학습률
                                # ↑ 빠른 수렴 가능, 발산 위험 ↑
                                # ↓ 안정성 ↑, 수렴 느려짐

        lr_scheduler={
                        'type': 'cosine',
                        'epoch_num': 2000,   # 총 학습 에폭 수(업데이트*epoch_per_collect) 근사치
                        'min_lr': 1e-5,
                        'min_lr_lambda': 0.1,      
                    },                       # 학습률 스케줄러 설정 (예: 'linear', 'cosine')
                                             # None이면 고정 학습률 사용, 스케줄러 사용 시 장기 학습 안정성 ↑

        value_weight=0.5,          # Value loss 가중치 (critic 손실 비중)
                                # ↑ 가치 함수 정확성 ↑, 정책 업데이트 속도 ↓
                                # ↓ 정책 업데이트 비중 ↑, 가치 추정 부정확 가능

        entropy_weight=0.01,       # 엔트로피 보너스 가중치 (탐험성)
                                # ↑ 탐험 ↑, 수렴 느려짐
                                # ↓ 탐험 ↓, 수렴 빨라짐 but 국소최적 해 위험 ↑

        clip_ratio=0.2,            # PPO 클리핑 비율 ε
                                # ↑ 정책 변화 폭 ↑, 발산 위험 ↑
                                # ↓ 정책 변화 폭 ↓, 안정성 ↑ but 수렴 느림

        adv_norm=True,             # Advantage 값 정규화 여부
                                # True → 업데이트 안정성 ↑
                                # False → 원본 값 사용, 스케일 변화에 민감

        value_norm=False,           # Value 타겟 정규화 여부
                                # True → 안정성 ↑, 값 스케일에 덜 민감
                                # False → raw value 사용

        ppo_param_init=True,       # PPO 권장 초기화 사용 여부
                                # True → 논문 권장 초기화로 학습 안정성 ↑

        grad_clip_type='clip_norm',# Gradient 클리핑 방식
                                # 'clip_norm' → 전체 norm 제한
                                # 'clip_value' → 값 자체 제한

        grad_clip_value=1,         # Gradient 클리핑 한계 값
                                # ↑ 크면 업데이트 폭 큼, 폭발 위험 ↑
                                # ↓ 안정성 ↑, 너무 작으면 업데이트 너무 미세

        ignore_done=False,         # done=True일 때 상태 리셋 여부
                                # True → 환경 종료 무시, 계속 학습
                                # False → 에피소드 종료 시 advantage 계산 초기화
    ),

    collect=dict(
        unroll_len=16,               # rollout 길이 (n-step 수집)
                                    # ↑ 길면 GAE 계산에 더 많은 미래 보상 반영, 메모리 사용 ↑
                                    # ↓ 길면 즉시 업데이트 가능 but 장기 보상 반영 ↓

        discount_factor=0.99,       # 감가율 γ
                                    # ↑ 미래 보상 중요도 ↑, 변동성 ↑
                                    # ↓ 즉시 보상 비중 ↑, 장기전략 학습 ↓

        gae_lambda=0.95,            # GAE(Generalized Advantage Estimation) 람다 값
                                    # ↑ 1.0에 가까우면 장기 의존성 ↑, 변동성 ↑
                                    # ↓ 단기 의존성 ↑, 안정적이나 장기 정보 손실
    ),

    other=dict(
        eps=dict(
            type='exp', start=1.0, end=0.05, decay=5000
        )
    )
)
cfg = EasyDict(config)

set_pkg_seed(0, use_cuda=cfg.cuda)
device = torch.device('cuda' if cfg.cuda else 'cpu')


model = VAC(**cfg.model).to(device)
policy = PPOPolicy(cfg, model=model)
writer = SummaryWriter(r"C:\Users\CIL\Desktop\DI-engine-main\unreal\tensorlog")

obs = env.reset()
obs_tensor = {aid: torch.tensor(obs[aid], dtype=torch.float32).unsqueeze(0).to(device) for aid in agent_ids}

transition_buffer = []
global_step = 0
epsilon = 1.0
epsilon_end = 0.05
epsilon_decay = 10000

episode_rewards_by_agent = {aid: [] for aid in agent_ids}
buffers = {aid: [] for aid in agent_ids}

episode_rewards_total = []

learn_count = 0  # 학습 호출 횟수 카운터

while global_step < TOTAL_STEPS:
    done = {aid: False for aid in agent_ids}
    done["__all__"] = False
    episode_reward = {aid: 0. for aid in agent_ids}
    episode_step = 0  

    alive_agents = set(agent_ids)

    while not done["__all__"]:
        with torch.no_grad():
            collect_output = policy.collect_mode.forward(obs_tensor)

        for aid in agent_ids:
            collect_output[aid]['logit'] = torch.nan_to_num(collect_output[aid]['logit'])
            collect_output[aid]['value'] = torch.nan_to_num(collect_output[aid]['value'])
        
        '''        
        actions = {}
        for aid in agent_ids:
            if random.random() < epsilon:
                actions[aid] = torch.tensor([env.action_space[aid].sample()], device=device)
            else:
                actions[aid] = collect_output[aid]['action']
        '''

        actions = {aid: collect_output[aid]['action'] for aid in agent_ids}
        logits = {aid: collect_output[aid]['logit'] for aid in agent_ids}
        values = {aid: collect_output[aid]['value'] for aid in agent_ids}

        # --- 환경 step ---
        step_result = env.step(actions)

        # 학습로그: 액션 + 보상
        rewards_debug = step_result['reward']
        print(
            f"[GLOBAL STEP {global_step}] [EPISODE STEP {episode_step}] [EPOCH {learn_count}] " +
            ", ".join([
                f"{aid}: action={env.valid_actions[actions[aid].item()]}, reward={rewards_debug[aid]:.2f}"
                for aid in agent_ids
            ])
        )

        next_obs_tensor = {
            aid: torch.nan_to_num(torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device))
            for aid, o in step_result['obs'].items()
        }
        done = step_result['done']

                
        '''
        #각 에이전트 obs 출력
        for aid in agent_ids:
            print(f"{aid} obs full: {obs_tensor[aid][0].cpu().numpy()}")
        '''

        # Step 업데이트
        global_step += 1
        episode_step += 1
        epsilon = max(epsilon_end, epsilon - (1.0 - epsilon_end) / epsilon_decay)

        # 중간 모델 저장
        if global_step % 500000 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = r"C:\Users\CIL\Desktop\DI-engine-main\unreal\learned_models"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"ppo_model_step{global_step}_{timestamp}.pth")
            torch.save(policy._state_dict_learn(), save_path)
            print(f"[AUTO-SAVED] Step {global_step} → {save_path}")

            # === 그래프 저장 디렉토리 추가 ===
            save_dir_plot = r"C:\Users\CIL\Desktop\DI-engine-main\unreal\learn_result"
            os.makedirs(save_dir_plot, exist_ok=True)

            #합산 보상만 사용
            if len(episode_rewards_total) >= 5:
                # 최근 10개 이동평균 시퀀스 생성
                avg_rewards_total = [
                    np.mean(episode_rewards_total[max(0, i-9):i+1])
                    for i in range(len(episode_rewards_total))
                ]

                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(avg_rewards_total) + 1), avg_rewards_total,
                        marker='o', markersize=4, alpha=0.7, label='Allies Total (10-ep avg)')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward (last 5 episodes)')
                plt.title(f'5-Episode Average Reward (Allies Total) (Step {global_step})')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                timestamp_plot = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(save_dir_plot, f"rewards_step{global_step}_{timestamp_plot}.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"[PLOT SAVED] {plot_path}")
            else:
                print("[PLOT] 에피소드가 10개 미만이라 중간 그래프 생성을 건너뜁니다.")

                
        # --- transition 생성: 에이전트별 버퍼에 시간순으로 저장 ---
        for aid in agent_ids:
            if aid not in alive_agents:
                continue  # 이미 죽은 에이전트는 이후 프레임 저장 안 함

            scaled_reward = step_result['reward'][aid] * reward_scale
            transition = {
                'obs':      obs_tensor[aid].squeeze(0).float(),
                'next_obs': next_obs_tensor[aid].squeeze(0).float(),
                'action':   actions[aid].squeeze(0).long(),                 # action은 long
                'logit':    logits[aid].squeeze(0).float(),
                'value':    values[aid].reshape(1).float(),
                'reward':   torch.tensor(scaled_reward, dtype=torch.float32),
                'done':     torch.tensor(1.0 if bool(done[aid]) else 0.0,   # ← bool → float32
                                        dtype=torch.float32),
                # 'traj_flag': torch.tensor(1.0 if bool(traj_flag[aid]) else 0.0, dtype=torch.float32),
            }
            buffers[aid].append(transition)
            episode_reward[aid] += scaled_reward

            # ★ 죽은 '그 스텝'까지는 저장하고, 그 다음부터는 배제
            if done[aid]:
                alive_agents.discard(aid)

        obs_tensor = next_obs_tensor

        # 에피소드 종료 처리
        if done["__all__"]:
            print(f"[EP DONE] Global Step: {global_step}, Rewards: {episode_reward}")
            for aid in agent_ids:
                episode_rewards_by_agent[aid].append(episode_reward[aid])

            total_reward = sum(episode_reward[aid] for aid in agent_ids)
            episode_rewards_total.append(total_reward)

            # TensorBoard 로그 추가
            writer.add_scalar("Reward/total", total_reward, global_step)
            for aid in agent_ids:
                writer.add_scalar(f"Reward/{aid}", episode_reward[aid], global_step)

            obs = env.reset()
            obs_tensor = {
                aid: torch.nan_to_num(torch.tensor(obs[aid], dtype=torch.float32).unsqueeze(0).to(device))
                for aid in agent_ids
            }

            # === 에피소드 종료 직후 그래프 저장 ===
            while global_step >= next_plot_step:
                save_dir_plot = r"C:\Users\CIL\Desktop\DI-engine-main\unreal\learn_result"
                os.makedirs(save_dir_plot, exist_ok=True)

                if len(episode_rewards_total) >= 10:
                    avg_rewards_total = [
                        np.mean(episode_rewards_total[max(0, i-9):i+1])
                        for i in range(len(episode_rewards_total))
                    ]

                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, len(avg_rewards_total) + 1), avg_rewards_total,
                             marker='o', markersize=4, alpha=0.7, label='Allies Total (10-ep avg)')
                    plt.xlabel('Episode')
                    plt.ylabel('Average Reward (last 10 episodes)')
                    plt.title(f'10-Episode Average Reward (Allies Total) (Step {global_step})')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()

                    plot_path = os.path.join(save_dir_plot, f"rewards_step{next_plot_step}.png")
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"[PLOT SAVED] {plot_path}")
                else:
                    print(f"[PLOT] 완료된 에피소드가 {len(episode_rewards_total)}개라 그래프 생성을 건너뜁니다 (필요: 10).")

                next_plot_step += PLOT_INTERVAL


    # --- 학습 조건 충족 시: 에이전트별 시퀀스 병합 → 학습 → 버퍼 클리어 ---
    if sum(len(buffers[aid]) for aid in agent_ids) >= 10000:
        all_tr = []
        for aid in agent_ids:
            all_tr.extend(buffers[aid])    # 각 에이전트의 연속 시퀀스를 보존한 채 병합
            buffers[aid].clear()           # on-policy 보장

        train_data = policy._get_train_sample(all_tr)

        # 학습에 쓰는 키와 dtype을 '강제'합니다. (그 외 키는 버립니다)
        _SCHEMA = {
            'obs':      torch.float32,
            'next_obs': torch.float32,
            'action':   torch.long,
            'logit':    torch.float32,
            'value':    torch.float32,
            'reward':   torch.float32,
            'done':     torch.float32,
            # DI-engine가 추가해 올 수 있는 키들을 쓰고 있다면 여기에 더 명시:
            # 'adv': torch.float32, 'return': torch.float32, 'weight': torch.float32,
        }

        def _to_tensor(v, dtype):
            # 1) 이미 텐서면 dtype만 강제
            if isinstance(v, torch.Tensor):
                return torch.nan_to_num(v.to(dtype))

            # 2) bool → float
            if isinstance(v, (bool, np.bool_)):
                return torch.tensor(float(v), dtype=dtype)

            # 3) 파이썬/넘파이 스칼라
            if isinstance(v, (int, float, np.integer, np.floating)):
                return torch.tensor(v, dtype=dtype)

            # 4) 넘파이 배열
            if isinstance(v, np.ndarray):
                if v.dtype == object:
                    # 객체 배열이면 각 원소를 수치화/스택
                    elems = []
                    for x in v:
                        if isinstance(x, torch.Tensor):
                            elems.append(x.detach().cpu().to(dtype))
                        elif isinstance(x, (np.ndarray, list, tuple, int, float, np.integer, np.floating, bool, np.bool_)):
                            elems.append(_to_tensor(x, dtype).cpu())
                        else:
                            raise TypeError(f'Unsupported ndarray element type: {type(x)}')
                    return torch.nan_to_num(torch.stack(elems).to(dtype))
                return torch.nan_to_num(torch.from_numpy(v).to(dtype))

            # 5) 리스트/튜플
            if isinstance(v, (list, tuple)):
                if len(v) > 0 and all(isinstance(x, torch.Tensor) for x in v):
                    return torch.nan_to_num(torch.stack([x.to(dtype) for x in v]))
                # 일반 수치/리스트 → 텐서
                try:
                    return torch.nan_to_num(torch.tensor(v, dtype=dtype))
                except Exception:
                    # 리스트 안에 섞인 타입들(텐서/넘파이 등)을 개별 변환 후 스택
                    elems = [_to_tensor(x, dtype) for x in v]
                    return torch.nan_to_num(torch.stack(elems))

            # 6) 마지막 시도: 넘파이로 캐스팅 후 처리
            try:
                arr = np.asarray(v)
                if arr.dtype == object:
                    raise TypeError(f'Unsupported object array from {type(v)}')
                return torch.nan_to_num(torch.from_numpy(arr).to(dtype))
            except Exception as e:
                raise TypeError(f'Unsupported type for training batch: {type(v)}') from e

        # 스키마 외 키 제거 + dtype/디바이스 강제
        _cleaned = []
        for t in train_data:
            nt = {}
            for k, dtype in _SCHEMA.items():
                if k in t:
                    nt[k] = _to_tensor(t[k], dtype).to(device)
            _cleaned.append(nt)

        train_data = _cleaned

        print(f"[LEARN-INPUT] B={len(train_data)} "
            f"obsμ={float(torch.stack([t['obs'] for t in train_data]).mean()):.2f} "
            f"rewμ={float(torch.stack([t['reward'] for t in train_data]).mean()):.2f} "
            f"valμ={float(torch.stack([t['value'] for t in train_data]).mean()):.2f} "
            f"done={float(torch.stack([t['done'] for t in train_data]).mean()):.2f}")

        # ← 추가: 배치 내 비정상 값 존재 시 학습 1회 스킵
        def _bad(td):
            for key in ('obs', 'logit'):
                if key in td and isinstance(td[key], torch.Tensor):
                    if not torch.isfinite(td[key]).all():
                        return True
            return False

        learn_output = None
        if any(_bad(t) for t in train_data):
            print("[GUARD] Non-finite values detected; skip this learn step.")
        else:
            with EnvKeepAlive(env, interval=0.3):
                learn_output = policy._forward_learn(train_data)

        learn_count += 1
        print(f"[LEARN] {learn_output}")

        if isinstance(learn_output, dict):
            for k, v in learn_output.items():
                try:
                    writer.add_scalar(f"Learn/{k}", float(v), global_step)
                except Exception:
                    pass
# 종료
env.close()
writer.close()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = r"C:\Users\CIL\Desktop\DI-engine-main\unreal\learned_models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"ppo_model_{timestamp}.pth")

torch.save(policy._state_dict_learn(), save_path)
print(f"[SAVED] {save_path}")

# === 보상 시각화: 최근 10개 평균 보상 (두 에이전트 합산만) ===
if len(episode_rewards_total) >= 10:
    avg_rewards_total = [
        np.mean(episode_rewards_total[max(0, i-9):i+1])
        for i in range(len(episode_rewards_total))
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_rewards_total) + 1), avg_rewards_total,
             marker='o', markersize=4, alpha=0.7, label='Allies Total (10-ep avg)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (last 10 episodes)')
    plt.title('10-Episode Average Reward (Allies Total) - Final')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_dir_plot = r"C:\Users\CIL\Desktop\DI-engine-main\unreal\learn_resert"
    os.makedirs(save_dir_plot, exist_ok=True)
    plot_path = os.path.join(save_dir_plot, f"rewards_final.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[PLOT SAVED] {plot_path}")
else:
    print("[WARN] 최소 10개 이상의 완료된 에피소드가 필요합니다.")