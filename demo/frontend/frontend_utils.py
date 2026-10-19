"""Translate slider values and send bounded requests to the inference service."""
import os
import requests

TASK_MAPPING = {"sen": "sentiment", "pol": "polite", "sta": "debate", "tox": "toxicity"}


def response_request(prompt, val_info) -> str:
    strengths = {}
    for task_key, strength in val_info.items():
        task_name = TASK_MAPPING[task_key]
        # Existing toxicity caches label the nontoxic class as positive.
        directed_strength = -strength if task_name == "toxicity" else strength
        if directed_strength:
            direction = "pos-neg" if directed_strength > 0 else "neg-pos"
            strengths[f"{task_name}_{direction}"] = abs(directed_strength)
    backend_url = os.getenv("NEUROSTEER_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        response = requests.post(
            f"{backend_url}/process",
            json={"input_dict": strengths, "input_str": prompt},
            timeout=(5, 180),
        )
        response.raise_for_status()
        result = response.json()["result"]
        if not isinstance(result, str):
            raise ValueError("Invalid result type")
        return result
    except requests.Timeout:
        return "请求超时。CPU 推理可能较慢，请稍后重试并检查后端日志。"
    except requests.ConnectionError:
        return f"无法连接推理服务（{backend_url}）。请先启动后端。"
    except requests.HTTPError:
        return f"推理服务暂不可用（HTTP {response.status_code}）。请查看后端 /health 和日志，确认模型及四组缓存齐全。"
    except (ValueError, KeyError):
        return "推理服务返回格式异常，请检查后端版本和日志。"
