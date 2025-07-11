#!/bin/bash
export WANDB_API_KEY="dce12064d30900b2cc538f73e82997de5aafbb96"
export HYDRA_FULL_ERROR=1
export RAY_RUNTIME_ENV_SKIP_WORKING_DIR_PACKAGING=1
python3 training/main_async_ppo.py \
    n_nodes=1 n_gpus_per_node=8 \
    allocation_mode=sglang.d4p1m1+d2p2m1 \
    cluster.fileroot=/home/szhangfa/AReaL/experiments \
    actor.type._class=qwen3 \
    actor.path=Qwen/Qwen3-1.7B \
    ref.type._class=qwen3 \
    ref.path=Qwen/Qwen3-1.7B \
    dataset.path=hf-dataset://inclusionAI/AReaL-RL-Data/data/boba_106k_0319.jsonl \
    dataset.train_bs_n_seqs=32 \
    group_size=8 \
    ppo.gen.max_new_tokens=4096 \
    ppo.ppo_n_minibatches=4 \
    actor_train.mb_spec.max_tokens_per_mb=32768 \
    actor_inf.mb_spec.max_tokens_per_mb=32768 \
    max_concurrent_rollouts=16 \
    max_head_offpolicyness=4 \
    wandb.mode=online \
    wandb.project=areal-async-ppo \
    wandb.name=async-ppo-boba-106k-0319

