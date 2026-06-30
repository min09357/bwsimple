# randread_bw — 메모리 읽기 대역폭 측정 도구 모음

시스템의 메모리 **read bandwidth(GB/s)** 를 측정합니다.
1GB 거대 페이지(hugepage) 영역에 대해 64B(캐시라인) 단위 read를 수행하며,
**무작위 접근**과 **순차 접근** 두 패턴을 동일한 하네스로 비교할 수 있습니다.

| 도구 | 패턴 | 설명 |
|---|---|---|
| `randread_bw` | 무작위 read | HPCC RandomAccess POLY LFSR 주소 생성, 캐시라인 단위 random read |
| `stream_bw` | 순차 read | 스레드별 연속 구간을 캐시라인 stride로 순차 스캔 (STREAM 류) |
| `sweep_bw.py` | — | 1코어부터 N코어까지 스윕하며 대역폭·이론 DRAM 피크 대비 % 출력 |
| `config.py` | — | DIMM·NUMA·hugepage·스윕 범위 등 모든 설정을 관리하는 설정 파일 |

## 설계 요점

| 항목 | 결정 |
|---|---|
| 메모리 영역 | `HUGEPAGES_1GB` × 1GB (`MAP_HUGE_1GB`), 연속 가상 주소 (기본 2GB) |
| 접근 단위 | 64B (AVX-512 aligned load = 1 캐시라인) |
| 연산 종류 | Read 전용 |
| 병렬화 | `std::jthread` + `std::barrier`, 코어당 1스레드 |
| 코어 핀 | `numactl -C <cpulist> -m <node>` 로 외부에서 설정 (`sweep_bw.py` 가 자동 처리) |
| DCE 방지 | 읽은 데이터를 `_mm512_xor` 누적 → checksum 출력 |

### randread_bw 전용

| 항목 | 결정 |
|---|---|
| 주소 생성 | HPCC POLY Galois LFSR, 스트림별 skip-ahead(`hpcc_starts`) |
| 스트림 수 | `ncores × 16` (스레드당 16개 독립 스트림 → MLP 극대화) |

#### 주소 충돌에 대하여
스트림마다 64비트 `ran` 수열 구간은 겹치지 않지만, 하위 25비트 인덱스가 같아 **캐시라인 주소가 코어 간·반복 간 겹칠 수 있습니다.**
이는 의도된 동작입니다. read 전용이라 coherence 비용이 없고, 작업집합이 일반적으로 L3(64MB)를 크게 초과하여 재방문해도 대부분 DRAM 접근으로 이어져 측정이 유효합니다.

### stream_bw 전용
영역을 코어 수만큼 연속 구간으로 분할하고, 각 스레드가 자기 구간을 캐시라인 stride로 순차 스캔합니다. `iters_per_thread`만큼 채울 때까지 구간을 wrap-around 반복합니다. 순차 접근이라 프리페처·row buffer 효과가 모두 살아 있어 `randread_bw`보다 훨씬 높은 대역폭이 나오는 것이 정상입니다.

## 빌드

```bash
make            # randread_bw, stream_bw 둘 다 빌드
```

요구 환경: g++ 13+, AVX-512F, Linux(1GB hugepage 지원).

## 실행

### 개별 벤치마크

```bash
# 무작위 read (기본값: 16코어, 1억회, 2GB 영역)
./randread_bw [ncores] [iters_per_thread] [hugepages_1gb]

# 순차 read (동일 인터페이스)
./stream_bw [ncores] [iters_per_thread] [hugepages_1gb]

# 예시: 8코어, 5천만회, 4GB 영역 (NUMA 인터리브 시스템에서는 numactl로 감싸 실행 권장)
numactl -C 0,2,4,6,8,10,12,14 -m 0 ./randread_bw 8 50000000 4
numactl -C 0,2,4,6,8,10,12,14 -m 0 ./stream_bw   8 50000000 4
```

- `ncores`: 사용할 코어 수 (기본 16). N > 16 이면 SMT 공유 경고.
- `iters_per_thread`: 스레드당 접근 횟수 (기본 1억). `randread_bw`는 UNROLL(16) 배수로 내림 조정됨.
- `hugepages_1gb`: 할당할 1GB hugepage 수 (기본 2). `randread_bw`는 **반드시 2의 거듭제곱**(1, 2, 4, 8, …)이어야 함 (LFSR 마스크 주소 방식 제약). `stream_bw`는 임의의 양수 가능.
- **주의**: CPU 번호가 NUMA 노드별로 인터리브된 시스템(예: node0=짝수, node1=홀수)에서 numactl 없이 실행하면 스레드가 여러 노드에 걸쳐 스케줄될 수 있어 측정이 오염됩니다. `sweep_bw.py`는 항상 numactl로 감싸 실행합니다.

### 코어 스윕 (sweep_bw.py)

1코어부터 `CORE_MAX`까지 코어 수를 늘려가며 대역폭과 **이론 DRAM 피크 대비 %** 를 막대그래프로 출력합니다.

```bash
python3 sweep_bw.py          # 무작위 접근 (기본)
python3 sweep_bw.py rand     # 무작위 접근
python3 sweep_bw.py stream   # 순차 접근
```

`config.py`에서 모든 설정을 관리합니다. 스크립트를 실행하기 전에 이 파일을 편집하세요.

| 항목 | 상수 | 의미 |
|---|---|---|
| DIMM | `DIMM_TRANSFER_RATE_MT_S` | DIMM 전송률 (예: DDR5-5200 → 5200) |
| DIMM | `DIMM_CHANNELS` | 채워진 메모리 채널 수 |
| NUMA | `NUMA_NODE` | 측정에 사용할 NUMA 노드 번호 (CPU affinity `-C` 및 메모리 `-m` 모두 이 노드로 고정) |
| Hugepage | `HUGEPAGES_1GB` | 할당할 1GB hugepage 수 (기본 2). rand 모드는 2의 거듭제곱 필수 |
| 스윕 | `CORE_START` / `CORE_MAX` / `CORE_STEP` | 스윕 코어 범위·증가폭 |
| 스윕 | `ITERS_PER_THREAD` | 바이너리에 넘길 스레드당 접근 횟수 |
| 기타 | `USE_SUDO` | 1GB hugepage 할당에 sudo가 필요하면 `True` |

이론 피크는 `전송률(MT/s) × 8 B × 채널 수 / 1000` 으로 계산합니다.
`sudo dmidecode -t memory | grep -E "Speed|Configured"` 로 실제 DIMM 값을 확인해 채우세요.

## 출력 해석

```
ncores=16  iters/thread=100000000  streams=256  region=2 GB
Warming up 2 GB ... done
```

`sweep_bw.py` 헤더 출력 예시:

```
=================================================================
 randread_bw — core sweep  [random access]
=================================================================
  DIMM rate   : 4800 MT/s × 8 B × 1 ch  →  peak 38.4 GB/s
  NUMA node   : 0  (CPUs: 0,1,2,3,...)
  Core range  : 1 .. 32  (step 1)
  Iters/thread: 100,000,000
  Hugepages   : 2 × 1GB  (2 GB region)
  Binary      : /path/to/randread_bw
=================================================================
```

바이너리 직접 실행 결과 예시:

```

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

- 실행 전 `grep HugePages /proc/meminfo` 로 `HugePages_Free >= HUGEPAGES_1GB` 확인.
  부족하면 `sudo sh -c 'echo N > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages'` 로 할당.
- `randread_bw`의 `hugepages_1gb`는 **2의 거듭제곱**만 허용 (1, 2, 4, 8, …). 그 외 값은 에러로 종료.
- 16GB 단일 채널 구성이면 DRAM 대역폭 상한이 낮아질 수 있음.
  `sudo dmidecode -t memory` 로 DIMM 수·속도 확인 권장.
