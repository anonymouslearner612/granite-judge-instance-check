#!/usr/bin/env python3
"""
Type your own messages and see what the Granite judge says about each one.

Run:   python try_prompt.py

Each message you type is sent through every check (prompt injection, toxicity, harmful
content, topic, intent, data exfiltration) and the verdict, P(yes) and time are printed.
Your earlier messages are kept as the conversation, so from the second message on the
chained_attack check also runs over the whole conversation.
Model settings, the topic/intent policy and YES_THRESHOLD come from the SETTINGS block
in instance_check.py.

Commands:
  /reset   forget the conversation and start a new one
  /quit    exit (Ctrl+C or Ctrl+D also work)
"""
import instance_check
from instance_check import CHECK_QUESTIONS, GraniteJudge, ensure_model_downloaded, load_model

# chained_attack needs earlier turns, so it only runs once there is a conversation.
MESSAGE_CHECKS = [check_name for check_name in CHECK_QUESTIONS if check_name != "chained_attack"]


def print_results(results):
    total_seconds = 0.0
    for result in results:
        flag = "FLAG" if result["verdict"] == "yes" else "ok"
        p_yes = "n/a" if result["p_yes"] is None else f"{result['p_yes']:.3f}"
        print(f"  {flag:<4}  {result['check']:<18}  P(yes)={p_yes}  {result['seconds']:.2f}s")
        total_seconds += result["seconds"]

    flagged_checks = [result["check"] for result in results if result["verdict"] == "yes"]
    if flagged_checks:
        print(f"  -> flagged by: {', '.join(flagged_checks)}   (total {total_seconds:.2f}s)")
    else:
        print(f"  -> nothing flagged   (total {total_seconds:.2f}s)")


def judge_message(judge, message, history):
    checks_to_run = list(MESSAGE_CHECKS)
    if history:
        checks_to_run.append("chained_attack")

    results = []
    for check_name in checks_to_run:
        results.append(judge.ask(check_name, message, history))
    return results


def main():
    print("loading model ...")
    llm = load_model(ensure_model_downloaded())
    judge = GraniteJudge(llm)

    print(f"allowed topics : {instance_check.DEFAULT_ALLOWED_TOPICS}")
    print(f"agent purpose  : {instance_check.DEFAULT_AGENT_PURPOSE}")
    print(f"yes threshold  : {instance_check.YES_THRESHOLD}")
    print("Type a message and press Enter. /reset starts a new conversation, /quit exits.\n")

    history = []
    while True:
        try:
            message = input(f"[turn {len(history) + 1}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        if message == "/quit":
            return
        if message == "/reset":
            history = []
            print("  conversation cleared\n")
            continue

        try:
            results = judge_message(judge, message, history)
        except ValueError as error:
            # llama.cpp raises ValueError when the conversation no longer fits CONTEXT_SIZE.
            print(f"  error: {error}\n  type /reset to start a new conversation\n")
            continue
        print_results(results)
        print()
        history.append({"role": "user", "content": message})


if __name__ == "__main__":
    main()
