# ETF를 활용한 글로벌 대체투자 자산들의 동적 연결성과 포트폴리오 최적화

윤병조(2024) 논문의 Python 구현

## 논문 정보
- **제목**: ETF를 활용한 글로벌 대체투자 자산들의 동적 연결성과 포트폴리오 최적화에 관한 연구
- **저자**: 윤병조
- **학술지**: 금융공학연구, 2024, 제23권 제1호, pp. 69-92

## 방법론
1. **TVP-VAR 모형** (Antonakakis, Chatziantoniou & Gabauer, 2020): Forgetting factor Kalman filter 기반 시변 파라미터 VAR
2. **Diebold-Yilmaz 연결성 분석** (2012, 2014): GFEVD 기반 동적 연결성 측정
3. **헤지비율**: Kroner & Sultan (1993) 최적 헤지비율
4. **포트폴리오 가중치**: Kroner & Ng (1998) 최적 포트폴리오 투자비중
5. **포트폴리오 최적화**: MVP (최소분산), MCP (최소상관), MCoP (최소연결성)
6. **헤징 효율성**: Ederington (1979)

## 분석 대상 ETF
| 티커 | 자산군 | 설명 |
|------|--------|------|
| USO  | 원유 | United States Oil Fund |
| VNQ  | 부동산 | Vanguard Real Estate ETF |
| IGF  | 인프라 | iShares Global Infrastructure ETF |
| ICLN | 친환경에너지 | iShares Global Clean Energy ETF |
| SPY  | 주식 | SPDR S&P 500 ETF |

## 분석 기간
2011년 1월 2일 ~ 2023년 9월 30일

## 실행 방법
```bash
pip install -r requirements.txt
python main.py
```

## 프로젝트 구조
```
├── main.py                  # 메인 분석 실행 스크립트
├── data_loader.py           # ETF 데이터 다운로드 및 전처리
├── tvp_var.py               # TVP-VAR 모형 (Kalman filter)
├── connectedness.py         # Diebold-Yilmaz 연결성 분석
├── hedge_ratio.py           # 헤지비율 및 포트폴리오 가중치
├── portfolio.py             # 포트폴리오 최적화 (MVP, MCP, MCoP)
├── visualization.py         # 시각화
├── requirements.txt         # 의존성 패키지
└── README.md
```
