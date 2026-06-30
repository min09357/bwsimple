# randread_bw — 메모리 읽기 대역폭 측정 도구 모음

시스템의 메모리 **read bandwidth(GB/s)** 를 측정합니다.
2GB 연속 1GB 거대 페이지(hugepage) 영역에 대해 64B(캐시라인) 단위 read를 수행하며,
**무작위 접근**과 **순차 접근** 두 패턴을 동일한 하네스로 비교할 수 있습니다.

| 도구 | 패턴 | 설명 |
|---|---|---|
| `randread_bw` | 무작위 read | HPCC RandomAccess POLY LFSR 주소 생성, 캐시라인 단위 random read |
| `stream_bw` | 순차 read | 스레드별 연속 구간을 캐시라인 stride로 순차 스캔 (STREAM 류) |
| `sweep_bw.py` | — | 1코어부터 N코어까지 스윕하며 대역폭·이론 DRAM 피크 대비 % 출력 |

## 설계 요점

| 항목 | 결정 |
|---|---|
| 메모리 영역 | 2GB (`MAP_HUGE_1GB` × 2장), 연속 가상 주소 |
| 접근 단위 | 64B (AVX-512 aligned load = 1 캐시라인) |
| 연산 종류 | Read 전용 |
| 병렬화 | `std::jthread` + `std::barrier`, 코어당 1스레드 |
| 코어 핀 | CPU 0..N-1 (Ryzen 9 7950X 기준 N≤16이면 물리코어) |
| DCE 방지 | 읽은 데이터를 `_mm512_xor` 누적 → checksum 출력 |

### randread_bw 전용

| 항목 | 결정 |
|---|---|
| 주소 생성 | HPCC POLY Galois LFSR, 스트림별 skip-ahead(`hpcc_starts`) |
| 스트림 수 | `ncores × 16` (스레드당 16개 독립 스트림 → MLP 극대화) |

#### 주소 충돌에 대하여
스트림마다 64비트 `ran` 수열 구간은 겹치지 않지만, 하위 25비트 인덱스가 같아 **캐시라인 주소가 코어 간·반복 간 겹칠 수 있습니다.**
이는 의도된 동작입니다. read 전용이라 coherence 비용이 없고, 작업집합(2GB)이 L3(64MB)의 ~32배라 재방문해도 대부분 DRAM 접근으로 이어져 측정이 유효합니다.

### stream_bw 전용
영역을 코어 수만큼 연속 구간으로 분할하고, 각 스레드가 자기 구간을 캐시라인 stride로 순차 스캔합니다. `iters_per_thread`만큼 채울 때까지 구간을 wrap-around 반복합니다. 순차 접근이라 프리페처·row buffer 효과가 모두 살아 있어 `randread_bw`보다 훨씬 높은 대역폭이 나오는 것이 정상입니다.

## 빌드

```bash
cd randread_bw
make            # randread_bw, stream_bw 둘 다 빌드
```

요구 환경: g++ 13+, AVX-512F, Linux(1GB hugepage 지원).

## 실행

### 개별 벤치마크

```bash
# 무작위 read (기본값: 16코어, 1억회)
./randread_bw [ncores] [iters_per_thread]

# 순차 read (동일 인터페이스)
./stream_bw [ncores] [iters_per_thread]

# 예시: 8코어, 5천만회
./randread_bw 8 50000000
./stream_bw   8 50000000
```

- `ncores`: 사용할 코어 수 (기본 16). N > 16 이면 SMT 공유 경고.
- `iters_per_thread`: 스레드당 접근 횟수 (기본 1억). `randread_bw`는 UNROLL(16) 배수로 내림 조정됨.
- 두 인자 모두 숫자만 입력. 대괄호 없이 위치 인자로 전달.

### 코어 스윕 (sweep_bw.py)

1코어부터 `CORE_MAX`까지 코어 수를 늘려가며 대역폭과 **이론 DRAM 피크 대비 %** 를 막대그래프로 출력합니다.

```bash
python3 sweep_bw.py          # 무작위 접근 (기본)
python3 sweep_bw.py rand     # 무작위 접근
python3 sweep_bw.py stream   # 순차 접근
```

스크립트 상단 상수로 측정 환경을 맞춥니다:

| 상수 | 의미 |
|---|---|
| `DIMM_TRANSFER_RATE_MT_S` | DIMM 전송률 (예: DDR5-5200 → 5200) |
| `DIMM_CHANNELS` | 채워진 메모리 채널 수 |
| `CORE_START` / `CORE_MAX` / `CORE_STEP` | 스윕 코어 범위·증가폭 |
| `ITERS_PER_THREAD` | 바이너리에 넘길 스레드당 접근 횟수 |
| `USE_SUDO` | 1GB hugepage 할당에 sudo가 필요하면 `True` |

이론 피크는 `전송률(MT/s) × 8 B × 채널 수 / 1000` 으로 계산합니다.
`sudo dmidecode -t memory | grep -E "Speed|Configured"` 로 실제 DIMM 값을 확인해 채우세요.

## 출력 해석

```
ncores=16  iters/thread=100000000  streams=256  region=2 GB
Warming up 2 GB ... done

=== Results ===
Elapsed        : 4.123 s
Total accesses : 1.600e+09  (1.024e+11 bytes)
Bandwidth      : 24.840 GB/s  (23.133 GiB/s)
GUPS           : 0.3880
Checksum       : 3f2a1b4c8d9e0f12
```

- **Bandwidth**: read 대역폭. `randread_bw`(무작위)는 `stream_bw`(순차)보다 훨씬 낮은 것이 정상 — 캐시라인 단위 무작위라 row buffer / 프리페처 효과가 없습니다.
- **GUPS**: 참고용. HPCC 정의와 달리 여기서는 read-only 기준.
- **Checksum**: DCE(dead-code elimination) 방지용 누적값. 같은 인자면 항상 동일해야 함.

## 주의사항

- 실행 전 `grep HugePages /proc/meminfo` 로 `HugePages_Free >= 2` 확인.
- 16GB 단일 채널 구성이면 DRAM 대역폭 상한이 낮아질 수 있음.
  `sudo dmidecode -t memory` 로 DIMM 수·속도 확인 권장.
