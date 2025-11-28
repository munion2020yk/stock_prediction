import FinanceDataReader as fdr
import yfinance as yf
from pykrx import stock
import pandas as pd
from datetime import datetime
import time

# --- 설정 ---
START_DATE = "2015-01-01"
END_DATE = datetime.now().strftime('%Y-%m-%d')
FILE_NAME = "KOSPI_base.csv"

print(f"데이터 수집 기간: {START_DATE} ~ {END_DATE}")

def get_kospi_data(start, end):
    print("1. KOSPI (OHLCV) 데이터 수집 중... (Source: FDR)")
    try:
        df = fdr.DataReader('KS11', start, end)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['KOSPI_Open', 'KOSPI_High', 'KOSPI_Low', 'KOSPI_Close', 'KOSPI_Volume']
        return df
    except Exception as e:
        print(f"  [오류] KOSPI 데이터 수집 실패: {e}")
        return pd.DataFrame()

def get_nasdaq_data(start, end):
    print("2. Nasdaq (OHLCV) 데이터 수집 중... (Source: Yfinance)")
    try:
        # yfinance 최신 버전 대응
        df = yf.download('^IXIC', start=start, end=end, progress=False)
        
        # MultiIndex 컬럼 처리 (Price, Ticker) -> Price만 남김
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Open', 'Close', 'Volume']]
        df.columns = ['NAS_Open', 'NAS_Close', 'NAS_Volume']
        return df
    except Exception as e:
        print(f"  [오류] Nasdaq 데이터 수집 실패: {e}")
        return pd.DataFrame()

def get_macro_indicators(start, end):
    print("3. 거시경제 지표 (환율, 유가, 금리, VKOSPI) 수집 중...")
    
    # 3-1. 환율 (USD/KRW) - FDR 우선, 실패 시 Yfinance
    try:
        usd_krw = fdr.DataReader('USD/KRW', start, end)[['Close']]
    except:
        usd_krw = yf.download('KRW=X', start=start, end=end, progress=False)[['Close']]
        if isinstance(usd_krw.columns, pd.MultiIndex): usd_krw.columns = usd_krw.columns.get_level_values(0)
    usd_krw.columns = ['USD_KRW_Close']
    
    # 3-2. 유가 (WTI)
    try:
        wti = fdr.DataReader('CL=F', start, end)[['Close']]
    except:
        wti = yf.download('CL=F', start=start, end=end, progress=False)[['Close']]
        if isinstance(wti.columns, pd.MultiIndex): wti.columns = wti.columns.get_level_values(0)
    wti.columns = ['WTI_Close']
    
    # 3-3. 금리 (Rate) - [수정됨]
    # KR3YT=RR (한국 3년물) 404 오류 발생 -> 미국 10년물 국채(^TNX)로 대체
    print("  -> 금리: 'KR3YT=RR' 대신 미국 10년물 국채금리('^TNX')를 수집합니다.")
    try:
        bond = yf.download('^TNX', start=start, end=end, progress=False)[['Close']]
        if isinstance(bond.columns, pd.MultiIndex): bond.columns = bond.columns.get_level_values(0)
    except Exception as e:
        print(f"  [오류] 금리 데이터 수집 실패: {e}")
        bond = pd.DataFrame()
    
    if not bond.empty:
        bond.columns = ['Rate']
    
    # 3-4. VKOSPI
    try:
        vkospi = fdr.DataReader('VKOSPI', start, end)[['Close']]
        vkospi.columns = ['VKOSPI_close']
    except:
        vkospi = pd.DataFrame()
    
    # 3-5. 선물 대용 (KOSPI 200)
    try:
        future = fdr.DataReader('KS200', start, end)[['Close']]
        future.columns = ['KOSPI_future']
    except:
        future = pd.DataFrame()

    # 병합
    dfs = [d for d in [usd_krw, wti, bond, vkospi, future] if not d.empty]
    if dfs:
        return pd.concat(dfs, axis=1)
    return pd.DataFrame()

def get_investor_data_pykrx(start, end):
    print("4. 투자자별 거래실적 (외국인) 수집 중... (Source: Pykrx)")
    start_str = start.replace('-', '')
    end_str = end.replace('-', '')
    
    try:
        # KOSPI 시장 외국인 순매수 금액
        df = stock.get_market_trading_value_by_date(start_str, end_str, "KOSPI")
        foreigner = df[['외국인']].copy()
        foreigner.columns = ['Foreign_rate']
        return foreigner
    except Exception as e:
        print(f"  [오류] Pykrx 데이터 수집 실패: {e}")
        return pd.DataFrame()

def main():
    # 1. 데이터 수집
    df_kospi = get_kospi_data(START_DATE, END_DATE)
    df_nasdaq = get_nasdaq_data(START_DATE, END_DATE)
    df_macro = get_macro_indicators(START_DATE, END_DATE)
    df_investor = get_investor_data_pykrx(START_DATE, END_DATE)
    
    print("5. 데이터 병합 및 CSV 저장 중...")
    
    # 데이터프레임 리스트 (비어있지 않은 것만)
    dfs = [d for d in [df_kospi, df_nasdaq, df_macro, df_investor] if not d.empty]
    
    if not dfs:
        print("수집된 데이터가 없습니다.")
        return

    # 전체 병합
    df_final = pd.concat(dfs, axis=1, join='outer')
    
    # 전처리: KOSPI 종가가 있는 날(한국 개장일)만 남김
    if 'KOSPI_Close' in df_final.columns:
        df_final = df_final[df_final['KOSPI_Close'].notna()]
    
    # 결측치 채우기 (Forward Fill)
    df_final.fillna(method='ffill', inplace=True)
    df_final.dropna(inplace=True) # 앞쪽 결측 제거
    
    df_final.index.name = 'Date'
    
    # CSV 저장 (탭 분리, UTF-8-SIG)
    # 기존 코드 호환성을 위해 sep='\t' 유지
    df_final.to_csv(FILE_NAME, sep='\t', encoding='utf-8-sig')
    
    print(f"\n[완료] '{FILE_NAME}' 파일 생성됨. (행 수: {len(df_final)})")
    print(df_final.tail())

if __name__ == "__main__":
    main()