# 커널 공부 저장소

[Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/) 5문제를 다시 풀어 보며 공부하는 저장소다.

## 문제

| 문제 | 내용 | 1위 (SGLang 대비) | 리더보드 |
|---|---|---:|---|
| `muse_bf16_proj` | 작은 m의 BF16/FP32 projection 5종 | 1.73× | [LEADERBOARD](okc-muse-spark/muse_bf16_proj/LEADERBOARD.md) |
| `muse_attention_decode` | decode attention | 3.34× | [LEADERBOARD](okc-muse-spark/muse_attention_decode/LEADERBOARD.md) |
| `muse_moe_w4a16` | NVFP4 가중치 MoE | 2.84× | [LEADERBOARD](okc-muse-spark/muse_moe_w4a16/LEADERBOARD.md) |
| `muse_decode_layer` | decode 레이어 전체 | 2.61× | [LEADERBOARD](okc-muse-spark/muse_decode_layer/LEADERBOARD.md) |
| `muse_prefill_layer` | prefill 레이어 전체 | 2.15× | [LEADERBOARD](okc-muse-spark/muse_prefill_layer/LEADERBOARD.md) |

1위와 리더보드는 [공식 리더보드](https://openkernelchallenge.dev/leaderboard)를 2026-11-09에 옮겨 적은 것이다. 운영진이 검토 후 최종 순위를 따로 발표하기로 해서 아직 최종이 아니다.

## 폴더

```
okc-muse-spark/
  bench.py, modal_baseline.py, modal_h200.py, MODAL.md   ← 공통 측정 하네스 (Modal H200, 공식 이미지 고정)
  benchmarks/2026-10-23-h200-baseline/                  ← 5문제 baseline을 H200에서 잰 기록
  <문제>/
    TASK.md, AGENTS.md, reference.py, baseline.py, checks.py, submission.py (, nvfp4.py)   ← 대회가 준 파일 (고치지 않는다)
    LEADERBOARD.md                                     ← 공식 리더보드 상위 10위 기록
```

대회가 준 파일은 고치지 않는다. 문제 하나를 커밋 하나(또는 몇 개)로 얹는다.

## 준비

```sh
uv sync                 # torch 등 설치
uv run modal setup      # Modal 인증 (H200 측정용, 한 번만)
```

## 공통 명령 (저장소 루트에서)

```sh
# H200: 문제 하나를 공식 채점 크기에서 baseline과 번갈아 측정 (결과는 modal_out/H200/<문제>/)
uv run modal run okc-muse-spark/modal_baseline.py --problem muse_bf16_proj \
  --submission submission.py --against baseline.py --repeats 15

# GPU 없이: 채점 shape 목록 보기
uv run python okc-muse-spark/bench.py --problem muse_attention_decode --suite scored --list-cases
```

측정 방식은 `okc-muse-spark/MODAL.md`에 있다.

## 이 저장소에 없는 것

- 문제 폴더 아래 `others/`는 `.gitignore`로 커밋에서 빠진다. 커밋하지 않을 비교용 파일을 둘 때 쓴다.
- `modal_out/`(측정 결과 JSON)은 git에 넣지 않는다.
