"""
Inference Script for Smart Waste Management System
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from smart_waste_management_system.client import (
    SmartWasteManagementSystemEnv,
)
from smart_waste_management_system.models import (
    SmartWasteManagementSystemAction,
)

IMAGE_NAME = os.getenv("IMAGE_NAME", "smart-waste-env:latest")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "smart_waste_management_system"

TASKS = [
    {"name": "waste_collection_easy",   "task_type": "easy"},
    {"name": "waste_collection_medium", "task_type": "medium"},
    {"name": "waste_collection_hard",   "task_type": "hard"},
]

MAX_STEPS = 20
TEMPERATURE = 0.3
MAX_TOKENS = 120

SUCCESS_SCORE_THRESHOLD = 0.6


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an intelligent waste collection planner.

    At each step, you must choose the index of the bin to collect next.

    Your goal is to:
    - Choose the bin index with the HIGHEST Fill level
    - Prefer closer bins (lower travel cost)
    - Minimize travel cost (traffic matters)
    - Avoid wasted trips (truck capacity is limited)
    - Avoid overflow at all costs

    Return ONLY a single integer representing the bin index.
    No explanation, no text.
    """
).strip()


# ---------------- Logging ---------------- #

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------- Prompt ---------------- #

def build_user_prompt(step: int, obs) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}

        Truck:
        - Position: {obs.truck_position}
        - Remaining Capacity: {obs.remaining_capacity:.2f}

        Bins:
        - Fill Levels: {obs.bin_fill_levels}
        - Fill Rates: {obs.bin_fill_rates}
        - Time Since Last Collection: {obs.time_since_last_collect}

        Environment:
        - Time of Day: {obs.time_of_day}
        - Traffic Level: {obs.traffic_level:.2f}
        - Peak Hours: {obs.peak_hours}

        Choose the best bin index to visit next.
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, obs) -> int:
    prompt = build_user_prompt(step, obs)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = (completion.choices[0].message.content or "").strip()

        # Extract integer safely
        return int(text)

    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return 0  # fallback

# ---------------- Single Task Run ---------------- #

async def run_task(env: SmartWasteManagementSystemEnv, client: OpenAI, task_name: str, task_type: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Pass task_type explicitly so the grader runs the correct difficulty
        result = await env.reset(task_type=task_type)
        obs = result.observation
        sum_of_task_scores = 0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_idx = get_model_action(client, step, obs)

            result = await env.step(
                SmartWasteManagementSystemAction(target_bin_index=action_idx)
            )

            # ADD THIS temporarily
            # print(f"[DEBUG] obs fields: {result.observation}", flush=True)
            print(f"[DEBUG] task_score: {result.observation.task_score}", flush=True)

            obs = result.observation
            reward = result.reward or 0.0
            done = obs.done
            sum_of_task_scores += result.observation.task_score

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=str(action_idx), reward=reward, done=done, error=None)

            # ✅ Read score BEFORE breaking
            # if obs.metadata and "score" in obs.metadata:
            #     score = float(obs.metadata["score"])
            #     print(f"[DEBUG] score from metadata: {score}", flush=True)

            if done:
                break

        score = max(0.01, min(sum_of_task_scores / 4.95, 0.99))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------- Main Loop ---------------- #

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await SmartWasteManagementSystemEnv.from_docker_image(IMAGE_NAME)

    try:
        for task in TASKS:
            await run_task(env, client, task_name=task["name"], task_type=task["task_type"])
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())