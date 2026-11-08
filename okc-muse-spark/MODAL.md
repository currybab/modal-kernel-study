# Muse Spark H200 실행

`modal_h200.py`는 H200 한 장에서 기존 `submission.py`를 검사하고 커널 시간을 측정한다. `bench.py`는 문제별 `reference.py`와 `checks.py`를 사용하며, 테스트 크기는 `TASK.md`에서 읽는다. 기존 커널과 문제 파일을 수정할 필요가 없다.

## 실행

저장소 루트에서 실행한다. Modal 인증이 없는 환경에서는 먼저 `uv run modal setup`을 실행한다. 이미지에는 CUDA 13.0 devel, Python 3.13, PyTorch 2.13.0, Triton 3.7.1과 C++ 빌드 도구가 포함된다. 첫 실행은 이미지 설치 시간이 추가된다.

```sh
# 작은 projection 사례 하나: 정확도 검사 + H200 시간 측정
uv run modal run okc-muse-spark/modal_h200.py

# 해당 문제의 공개 테스트 전부 검사
uv run modal run okc-muse-spark/modal_h200.py --problem muse_attention_decode --mode check --suite tests

# 제출 커널을 실제 채점 크기에서 측정
uv run modal run okc-muse-spark/modal_h200.py --problem muse_bf16_proj --suite scored --repeats 10

# 특정 크기만 반복 측정
uv run modal run okc-muse-spark/modal_h200.py --problem muse_attention_decode --suite scored --case-id 0 --repeats 10

# 다른 구현 파일 비교: 해당 문제 폴더 안의 파일명
uv run modal run okc-muse-spark/modal_h200.py --problem muse_bf16_proj --submission submission_triton.py

# GPU를 띄우지 않고 사례 번호 확인
uv run python okc-muse-spark/bench.py --problem muse_attention_decode --suite scored --list-cases
```

`okc-muse-spark` 폴더 안에서는 `uv run modal run modal_h200.py ...`로 실행해도 된다.

| 옵션 | 값과 기본값 |
|---|---|
| `--problem` | 기본 `muse_bf16_proj`; `muse_attention_decode`, `muse_moe_w4a16`, `muse_decode_layer`, `muse_prefill_layer` 지원 |
| `--mode` | `check`: 정확도만 검사, `bench`: 정확도 검사 후 측정. 기본 `bench` |
| `--suite` | `smoke`: 첫 공개 테스트 하나, `tests`: 공개 테스트 전부, `scored`: 문서의 채점 크기. 기본 `smoke` |
| `--case-id` | 사례 번호 하나 선택. 기본 `-1`은 선택한 suite 전체 |
| `--submission` | 문제 폴더 기준 구현 파일 경로. 기본 `submission.py` |
| `--repeats` | 측정 반복 횟수. 기본 5. `check` 모드에는 적용하지 않음 |
| `--seed` | 재현 가능한 입력 seed. 기본 2026 |

실행 결과는 저장소의 `modal_out/H200/<문제>/<시각>-<모드>.json`에 저장된다. GPU와 라이브러리 버전, 제출 파일 해시, 사례별 검사 결과, 반복별 시간과 중앙값·최솟값·최댓값이 포함된다. 검사나 실행이 실패하면 명령도 실패하며, 완료한 사례와 오류를 결과 파일에 남긴다. 전체 layer와 긴 prefill 크기는 상당한 실행 시간이 필요할 수 있어 첫 실험에는 `smoke`나 `--case-id`를 사용한다.

## 측정 방식

가중치를 생성하고 `prepare(weights)`를 한 번 호출한 뒤, 각 크기에서 `kernel_fn`을 워밍업하고 CUDA graph로 캡처한다. 매 반복마다 다른 seed로 입력을 생성해 캡처된 버퍼에 복사한다. 128 MiB 버퍼를 읽어 L2를 비운 다음, 그래프 내부 CUDA event 사이의 커널 실행 시간을 마이크로초로 측정한다. 입력 생성·복사, 캐시 비우기, 정확도 검사, 컴파일과 `prepare` 시간은 측정에서 제외한다. 모든 측정 반복의 출력을 검사하며 attention과 layer 문제는 새 KV cache 값도 검사한다. graph 캡처에 실패하면 eager 시간으로 대체하지 않고 오류를 보고한다.

이 결과는 개발용 절대 시간이다. 공식 OKC의 SGLang 교대 측정, paired speedup, 신뢰구간이나 공식 점수를 재현하지 않는다. 전체 layer의 공식 router 부하 목표도 `TASK.md`의 표에 없으므로 보정하지 않는다. 문서에 공개된 크기와 reference 검사를 이용해 구현을 비교할 때 사용한다.

기본 `modal_h200.py` 이미지에는 `sglang`과 `sgl_kernel`이 없다. 제공된 starter 및 PyTorch/Triton/CUDA 구현을 실행할 수 있으며, reference의 선택적 FA3 scheduler metadata는 해당 패키지가 없으면 `None`이다. SGLang baseline은 아래의 전용 환경으로 실행한다.

## SGLang baseline 측정

첫 기준 측정은 [2026년 10월 23일 H200 baseline 기록](benchmarks/2026-10-23-h200-baseline/README.md)에 원본 결과와 함께 보관한다.

`modal_baseline.py`는 공식 `lmsysorg/sglang:dev-muse-spark-open` 이미지의 고정 digest와 baseline에 명시된 SGLang 커밋 `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`의 Python/JIT 소스를 사용한다. FA3와 Marlin을 포함한 공식 이미지의 native 의존성을 유지한다. `baseline.py` 자체는 변경하지 않는다. 기본 starter 이미지와 Torch 버전이 다를 수 있으므로 결과의 `environment`도 함께 비교한다.

```sh
# 다섯 문제의 채점 크기를 순서대로 측정 (기본 반복 5회)
uv run modal run okc-muse-spark/modal_baseline.py

# 문제 하나만 측정
uv run modal run okc-muse-spark/modal_baseline.py --problem muse_bf16_proj --repeats 5

# 작은 사례로 환경 확인
uv run modal run okc-muse-spark/modal_baseline.py --problem muse_attention_decode --suite smoke
```

결과는 `modal_out/H200/<문제>/<시각>-baseline-bench.json`에 저장한다. `scored` 전체를 성공한 실행에는 `aggregate.loads_ms`와 `aggregate.geomean_ms`를 추가한다. 부하별 시간은 각 사례의 중앙값에 문서의 `weight`를 곱해 합산한 값이며, 문제 전체 시간은 부하별 시간의 기하평균이다. 사례를 `--case-id`로 일부만 선택하면 문제 전체 시간을 계산하지 않는다. 이 합산은 공식 리더보드 점수나 paired speedup을 대신하지 않는다.

`--problem`에는 문제 이름 여러 개를 쉼표로 연결해 지정할 수도 있다. 문제 하나가 실패해도 나머지 문제는 계속 실행하며, 마지막에 실패 여부를 반환한다.

## 공식 검사와 제출

공식 `okc check`와 `okc submit`은 OKC 서버에서 실행된다. Modal GPU를 띄울 필요 없이 로컬에서 실행하며, Modal 러너는 자동 제출하지 않는다. 최초 `okc login`과 규칙 동의는 사용자가 직접 진행한다.

```sh
cd okc-muse-spark/muse_attention_decode
okc check submission.py --json
okc submit submission.py --json
```

## 참고

[GPU MODE Modal runner](https://github.com/gpu-mode/kernelbot/blob/main/src/runners/modal_runner.py)의 컨테이너 구성과 [평가 실행기](https://github.com/gpu-mode/kernelbot/blob/main/examples/eval.py)의 검사·측정 분리를 참고했다. 실제 함수 인터페이스와 정확도 검사는 이 폴더의 OKC reference를 따른다. GPU 지정은 [Modal GPU 문서](https://modal.com/docs/guide/gpu)를 참고한다.
