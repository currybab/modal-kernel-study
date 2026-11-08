# Muse Spark SGLang baseline H200 측정 기록

2026년 10월 23일 Modal H200 한 장에서 기존 `baseline.py`를 변경하지 않고 측정했다. `scored` 68개 크기의 각 5회 측정, 총 340회 재생이 모두 reference 정확도 검사를 통과했다. 이 기록을 이후 Modal 실험의 초기 비교 기준으로 사용한다.

## 결과

단위는 ms다. 문제 전체 시간은 각 부하에서 `사례 중앙값 × weight`를 합산한 뒤 부하 간 기하평균을 낸 값이다. 차이는 `이번 측정 / 공식 기준 − 1`이며, 최적화 성능 향상률을 뜻하지 않는다.

| 문제 | 공식 SGLang 기준 | Modal 측정 | 차이 | 채점 크기 수 |
|---|---:|---:|---:|---:|
| muse_bf16_proj | 4.8308 | 6.4519 | +33.6% | 30 |
| muse_attention_decode | 3.7663 | 4.2403 | +12.6% | 12 |
| muse_moe_w4a16 | 13.4485 | 13.2456 | −1.5% | 6 |
| muse_decode_layer | 22.6203 | 23.4755 | +3.8% | 12 |
| muse_prefill_layer | 387.4039 | 343.1877 | −11.4% | 8 |

공식 기준값은 당시 `okc show <문제> --json`의 `sglang_us`에서 가져왔다. [공식 메타데이터 원본](official-baseline-metadata.json)과 [요약 JSON](summary.json), [요약 CSV](summary.csv)를 함께 보관한다. 요약 JSON에는 사용자가 전달한 당시 리더보드 최고 시간도 출처를 구분해 기록했다.

## 실행 조건

| 항목 | 값 |
|---|---|
| GPU | NVIDIA H200, 1장 |
| 실행기 | `modal_baseline.py` → `bench.py` |
| Python | 3.12.3 |
| PyTorch | 2.14.1+cu130 |
| CUDA runtime | 13.0 |
| Triton | 3.8.0 |
| sglang-kernel | 0.4.9 |
| SGLang 소스 | `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f` |
| 입력 seed 옵션 | 2026 |
| 반복 횟수 | 크기별 5회 |
| CPU 및 메모리 요청 | 8 CPU, 65536 MiB |

이미지는 `lmsysorg/sglang@sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af`다. 공식 Muse Spark 이미지의 native 의존성을 유지하고 Python/JIT 소스를 위 커밋으로 고정했다. 실행기와 문제별 Python 소스의 SHA-256은 [소스 manifest](source-manifest.json)에 있다. `git_head_before_archive`는 기록 전 커밋이며, 측정 실행기의 정확한 내용은 manifest의 파일 해시로 확인한다.

## 측정 및 검증 방법

가중치는 각 reference의 `WEIGHTS_SEED`로 생성한다. `prepare`는 한 번, 크기별 워밍업은 세 번 실행하고 CUDA graph를 캡처한다. 각 크기의 기준 seed는 `(2026 + case_id × 100003) mod 2^63`, 측정 반복은 여기에 1부터 5를 더한 seed를 사용한다. 입력 텐서는 매 반복 새로 생성하고 캡처된 버퍼에 복사한다.

각 graph 재생에서 128 MiB 버퍼를 읽어 L2를 비운 뒤, graph 내부 CUDA event로 `kernel_fn` 실행 구간을 측정한다. 입력 생성·복사, 캐시 비우기, 컴파일, `prepare`, 정확도 검사는 시간에서 제외한다. 최초 eager 실행과 모든 측정 재생을 reference로 검증하며, attention과 layer는 KV cache 갱신도 검사한다. 검증 범위는 `scored` 크기이고, 공개 `tests` suite 전체를 실행한 기록은 아니다.

## 해석과 후속 비교

공식 심사기의 graph replay 고정비용 차감, baseline과 후보의 교대 측정, paired speedup 및 신뢰구간, router 부하 목표 보정은 적용하지 않았다. 공식 기준과 차이가 나는 원인은 이 측정만으로 분리할 수 없다. 이 값에서 공식 리더보드 대비 speedup을 계산하지 않는다.

후속 후보는 같은 이미지, GPU 종류, 입력 및 가중치 seed, 크기와 반복 수로 baseline과 함께 다시 측정한다. 초기 기록과의 비교만으로 향상을 확정하지 않는다. 특히 기본 `modal_h200.py` 환경은 Torch 2.13.0·Triton 3.7.1이므로 이 baseline 환경과 버전이 다르다.

attention의 case 5는 5개 샘플이 약 392–518µs로 퍼져 있었다. 입력별 길이와 실행 조건의 영향을 분리하지 않았으며, 해당 크기 최적화는 더 많은 반복과 baseline 교대 측정으로 확인한다.

## 재현

저장소 루트에서 다음을 실행한다. 처음 실행하는 환경은 `uv run modal setup`이 필요하다.

```sh
uv run modal run okc-muse-spark/modal_baseline.py --problem all --suite scored --mode bench --repeats 5 --seed 2026
```

실제 측정은 projection을 먼저 실행한 뒤 나머지 네 문제를 순서대로 실행했다. 해당 실행 링크는 [projection](https://modal.com/apps/nerdface-ai/main/ap-zKyp16840I19qO62Lep2RJ)과 [attention·MoE·decoder·prefill](https://modal.com/apps/nerdface-ai/main/ap-uCDxV6Kh9nhrOfvdadzSBL)이다. 재실행은 새 결과를 `modal_out/H200/`에 저장하며 이 기록을 덮어쓰지 않는다.

## 원본 결과

- [Projection 30개 크기](muse_bf16_proj.json)
- [Attention 12개 크기](muse_attention_decode.json)
- [MoE 6개 크기](muse_moe_w4a16.json)
- [Decoder layer 12개 크기](muse_decode_layer.json)
- [Prefill layer 8개 크기](muse_prefill_layer.json)

각 파일에는 크기별 입력 조건, 검증 메시지, 반복별 µs 시간, 중앙값·최솟값·최댓값, 환경 및 baseline 파일 해시가 들어 있다.
