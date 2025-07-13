#!/bin/bash
# Example script to train PPO on the Personality Alignment dataset.

export HYDRA_FULL_ERROR=1

python3 training/main_async_ppo.py \
    actor.type._class=qwen3 \
    actor.path=Qwen/Qwen3-1.7B \
    ref.type._class=qwen3 \
    ref.path=Qwen/Qwen3-1.7B \
    dataset.path=Personality-Alignment/dialogue_dataset.jsonl \
    dataset.max_prompt_len=1024 \
    dataset.train_bs_n_seqs=32 \
    group_size=8 \
    ppo.gen.max_new_tokens=512 \
    ppo.ppo_n_minibatches=4

