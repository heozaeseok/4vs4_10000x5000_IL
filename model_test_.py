# === ppo_eval_min_print.py ===
import os, torch
from easydict import EasyDict
from ding.model import VAC
from ding.policy import PPOPolicy
from multi_agent_env import SocketMultiAgentEnv

MODEL_PATH = r"C:\Users\CIL\Desktop\DI-engine-main\unreal\learned_models\10000x5000_4vs4_ver6.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPISODES = 50
GREEDY = False

env = SocketMultiAgentEnv({
    'map_size': 5000,
    'max_step': 300,
    'win_reward': 100,
    'num_detectable': 4,  # 적 4명 감지
    'num_agents': 4       # 아군 4명
})

agent_ids = env.agent_ids
obs_dim = env.observation_space[agent_ids[0]].shape[0]
act_dim = env.action_space[agent_ids[0]].n

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


model = VAC(**cfg.model).to(DEVICE)
policy = PPOPolicy(cfg, model=model)

state = torch.load(MODEL_PATH, map_location=DEVICE)
if 'learn_model' in state:
    sd = state['learn_model']
    policy._learn_model.load_state_dict(sd)
    policy._collect_model.load_state_dict(state.get('collect_model', sd))
    policy._eval_model.load_state_dict(state.get('eval_model', sd))
elif 'model' in state:
    sd = state['model']
    policy._learn_model.load_state_dict(sd)
    policy._collect_model.load_state_dict(sd)
    policy._eval_model.load_state_dict(sd)
policy._learn_model.eval(); policy._collect_model.eval(); policy._eval_model.eval()

episodes = 0
with torch.inference_mode():
    while episodes < NUM_EPISODES:
        ts = env.reset()
        obs_tensor = {aid: torch.as_tensor(ts[aid], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                      for aid in agent_ids}
        done_all, step = False, 0
        while not done_all and step < env.max_step:
            out = (policy.eval_mode.forward(obs_tensor)
                   if GREEDY else policy.collect_mode.forward(obs_tensor))
            actions = {}
            for aid in agent_ids:
                oi = out.get(aid, {})
                if 'action' in oi:
                    a = oi['action']
                elif 'logit' in oi:
                    a = torch.argmax(oi['logit'], dim=-1, keepdim=False).view(1)
                else:
                    a = torch.zeros(1, dtype=torch.long, device=DEVICE)
                actions[aid] = a

            # ★ 스텝마다 행동 출력 ★
            act_str = ", ".join([f"{aid}:{int(actions[aid].item())}" for aid in agent_ids])
            print(f"[Step {step+1}] {act_str}")

            step_result = env.step(actions)
            obs_tensor = {aid: torch.as_tensor(step_result['obs'][aid], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                          for aid in agent_ids}
            done_all = bool(step_result['done'].get("__all__", False))
            step += 1
        episodes += 1

env.close()
