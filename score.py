import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
import sys
import os
import json

# 지도 시각화를 위한 추가 import
try:
    import osmnx as ox
    import networkx as nx
    from geopy.distance import geodesic
    import folium
    import requests
    import webbrowser
    from dotenv import load_dotenv
    from scipy.spatial import cKDTree
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("시각화 라이브러리가 설치되어 있지 않습니다. 위험도 분석만 수행됩니다.")

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    import locale
    try:
        os.system('chcp 65001 > nul')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

if VISUALIZATION_AVAILABLE:
    load_dotenv()  # env 파일에서 Kakao API key 불러오기

class PedestrianRiskPredictor:
    def __init__(self):
        try:
            self.data = pd.read_csv('교통사고다발구역.csv', encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.data = pd.read_csv('교통사고다발구역.csv', encoding='cp949')
            except UnicodeDecodeError:
                self.data = pd.read_csv('교통사고다발구역.csv', encoding='euc-kr')
        self.model = None
        self.scaler = MinMaxScaler()
        self.prepare_data()


    def haversine_distance(self, lat1, lon1, lat2, lon2):
        try:
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371
            return c * r
        except:
            return float('inf')

    def prepare_data(self):
        print("데이터 전처리 시작...")

        self.data = self.data.dropna(subset=['X', 'Y'])

        self.data['X'] = pd.to_numeric(self.data['X'], errors='coerce')
        self.data['Y'] = pd.to_numeric(self.data['Y'], errors='coerce')
        self.data['OCCU_TM'] = pd.to_numeric(self.data['OCCU_TM'], errors='coerce')

        self.data = self.data.dropna(subset=['X', 'Y', 'OCCU_TM'])

        severity_scores = {
            '경상': 1,
            '부상신고': 2,
            '중상': 4,
            '사망': 10
        }
        self.data['severity_score'] = self.data['LCLAS'].map(severity_scores).fillna(1)

        self.data['WLKG'] = self.data['WLKG'].astype(str)
        self.data['SCLAS'] = self.data['SCLAS'].astype(str)

        pedestrian_condition = (
            (self.data['SCLAS'].str.contains('차대사람', na=False)) |
            (self.data['WLKG'].str.contains('O', na=False))
        )
        self.data['pedestrian_weight'] = np.where(pedestrian_condition, 3, 1)

        def get_time_period(hour):
            try:
                hour = int(hour)
                if 0 <= hour < 6:
                    return '새벽'
                elif 6 <= hour < 12:
                    return '오전'
                elif 12 <= hour < 18:
                    return '오후'
                else:
                    return '저녁'
            except:
                return '오전'

        self.data['time_period'] = self.data['OCCU_TM'].apply(get_time_period)

        weekend_days = ['토요일', '일요일']
        self.data['is_weekend'] = self.data['OCCU_DAY'].isin(weekend_days).astype(int)

        print(f"전체 사고 데이터: {len(self.data)}건")
        print(f"보행자 관련 사고: {len(self.data[self.data['pedestrian_weight'] == 3])}건")

        print("\n데이터 샘플:")
        print(self.data[['LCLAS', 'SCLAS', 'WLKG', 'severity_score', 'pedestrian_weight']].head())

    def calculate_risk_features(self, target_lat, target_lon, radius_km=0.5):
        
        try:
            distances = self.data.apply(
                lambda row: self.haversine_distance(target_lat, target_lon, row['Y'], row['X']),
                axis=1
            )
            nearby_accidents = self.data[distances <= radius_km].copy()

            if len(nearby_accidents) == 0:
                return {
                    'total_accidents': 0,
                    'pedestrian_accidents': 0,
                    'severity_weighted_score': 0,
                    'pedestrian_weighted_score': 0,
                    'weekend_accidents': 0,
                    'night_accidents': 0,
                    'fatal_accidents': 0
                }

            night_condition = (
                (nearby_accidents['OCCU_TM'] >= 22) |
                (nearby_accidents['OCCU_TM'] <= 6)
            )
            night_accidents = len(nearby_accidents[night_condition])

            fatal_condition = nearby_accidents['LCLAS'].str.contains('사망', na=False)
            fatal_accidents = len(nearby_accidents[fatal_condition])

            features = {
                'total_accidents': len(nearby_accidents),
                'pedestrian_accidents': len(nearby_accidents[nearby_accidents['pedestrian_weight'] == 3]),
                'severity_weighted_score': (nearby_accidents['severity_score'] * nearby_accidents['pedestrian_weight']).sum(),
                'pedestrian_weighted_score': nearby_accidents['pedestrian_weight'].sum(),
                'weekend_accidents': nearby_accidents['is_weekend'].sum(),
                'night_accidents': night_accidents,
                'fatal_accidents': fatal_accidents
            }

            return features

        except Exception as e:
            print(f"특성 계산 중 오류: {e}")
            return {
                'total_accidents': 0,
                'pedestrian_accidents': 0,
                'severity_weighted_score': 0,
                'pedestrian_weighted_score': 0,
                'weekend_accidents': 0,
                'night_accidents': 0,
                'fatal_accidents': 0
            }

    def create_training_data(self, n_samples=800):
        
        print("훈련 데이터 생성 중...")

        sample_size = min(n_samples // 2, len(self.data))
        sample_indices = np.random.choice(len(self.data), sample_size, replace=False)
        sample_locations = self.data.iloc[sample_indices][['Y', 'X']].values

        lat_min, lat_max = self.data['Y'].min(), self.data['Y'].max()
        lon_min, lon_max = self.data['X'].min(), self.data['X'].max()

        additional_samples = n_samples - len(sample_locations)
        if additional_samples > 0:
            random_lats = np.random.uniform(lat_min, lat_max, additional_samples)
            random_lons = np.random.uniform(lon_min, lon_max, additional_samples)
            random_locations = np.column_stack([random_lats, random_lons])
            sample_locations = np.vstack([sample_locations, random_locations])

        training_features = []
        training_targets = []

        for i, (lat, lon) in enumerate(sample_locations):
            if i % 100 == 0:
                print(f"진행률: {i}/{len(sample_locations)}")

            features = self.calculate_risk_features(lat, lon)

            import math

            # 더 세밀한 가중치 적용으로 점수 분포를 개선
            total_weight = (
                features['total_accidents'] * 0.05 +
                features['pedestrian_accidents'] * 0.3 +
                features['fatal_accidents'] * 0.8 +
                features['night_accidents'] * 0.1 +
                features['weekend_accidents'] * 0.05
            )

            # 심각도 가중치를 낮게 조정
            severity_factor = features['severity_weighted_score'] * 0.02

            # 최종 점수 계산 (1-10 범위로 더 부드럽게 분배)
            raw_score = 1.0 + total_weight + severity_factor

            # 점수를 1-10 범위로 스케일링 (로그 함수로 부드럽게)
            if raw_score > 1:
                risk_score = 1.0 + math.log(raw_score) * 1.8
            else:
                risk_score = 1.0

            risk_score = max(1.0, min(10.0, risk_score))

            training_features.append(list(features.values()))
            training_targets.append(risk_score)

        return np.array(training_features), np.array(training_targets)

    def train_model(self, n_samples=800):
        
        print("모델 훈련 시작...")

        try:
            X, y = self.create_training_data(n_samples)

            X_scaled = self.scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)

            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)

            print(f"훈련 점수: {train_score:.4f}")
            print(f"테스트 점수: {test_score:.4f}")
            print("모델 훈련 완료!")

        except Exception as e:
            print(f"모델 훈련 중 오류 발생: {e}")
            raise

    def predict_risk(self, latitude, longitude):
        
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다. train_model()을 먼저 실행하세요.")

        try:
            features = self.calculate_risk_features(latitude, longitude)

            X_pred = np.array([list(features.values())])
            X_pred_scaled = self.scaler.transform(X_pred)
            risk_score = self.model.predict(X_pred_scaled)[0]

            # 점수를 적절한 범위로 분배
            risk_score = max(1, min(10, risk_score))

            return {
                'risk_score': round(risk_score, 1),
                'risk_level': self.get_risk_level(risk_score),
                'nearby_accidents': features['total_accidents'],
                'pedestrian_accidents': features['pedestrian_accidents'],
                'details': features
            }

        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            return {
                'risk_score': 1.0,
                'risk_level': "매우 낮음",
                'nearby_accidents': 0,
                'pedestrian_accidents': 0,
                'details': {}
            }

    def get_risk_level(self, score):
        
        if score <= 2:
            return "매우 낮음"
        elif score <= 4:
            return "낮음"
        elif score <= 6:
            return "보통"
        elif score <= 8:
            return "높음"
        else:
            return "매우 높음"

    def analyze_rectangular_area(self, lat_a, lon_a, lat_b, lon_b, grid_distance_m=75):
        
        import json

        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다. train_model()을 먼저 실행하세요.")

        center_lat = (lat_a + lat_b) / 2
        center_lon = (lon_a + lon_b) / 2

        half_lat = abs(lat_a - lat_b) / 2
        half_lon = abs(lon_a - lon_b) / 2

        min_lat = center_lat - half_lat * 3
        max_lat = center_lat + half_lat * 3
        min_lon = center_lon - half_lon * 3
        max_lon = center_lon + half_lon * 3

        print(f"분석 지역: ({min_lat:.6f}, {min_lon:.6f}) ~ ({max_lat:.6f}, {max_lon:.6f})")

        lat_step = grid_distance_m / 111000
        avg_lat = (min_lat + max_lat) / 2
        lon_step = grid_distance_m / (111000 * np.cos(np.radians(avg_lat)))

        print(f"격자 간격: 위도 {lat_step:.6f}도, 경도 {lon_step:.6f}도")

        lat_points = int((max_lat - min_lat) / lat_step) + 1
        lon_points = int((max_lon - min_lon) / lon_step) + 1
        total_points = lat_points * lon_points

        print(f"총 {total_points}개 격자점 분석 예정")

        danger_centers = []
        grid_count = 0

        lat = min_lat
        while lat <= max_lat:
            lon = min_lon
            while lon <= max_lon:
                result = self.predict_risk(lat, lon)

                weight = round(result['risk_score'], 1)

                danger_centers.append({
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "weight": weight
                })

                grid_count += 1
                if grid_count % 50 == 0:
                    print(f"진행률: {grid_count}/{total_points} 격자점 처리 완료 ({grid_count/total_points*100:.1f}%)")

                lon += lon_step
            lat += lat_step

        print(f"총 {len(danger_centers)}개 격자점 분석 완료")

        danger_centers.sort(key=lambda x: (x['lat'], x['lon']))

        return danger_centers

    def save_danger_centers_json(self, danger_centers, filename="danger_centers_output.json"):
        
        import json

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(danger_centers, f, ensure_ascii=False, indent=4)

        print(f"결과가 {filename} 파일로 저장되었습니다.")

def main():
    try:
        print("보행자 위험도 예측 시스템을 시작합니다...")
        predictor = PedestrianRiskPredictor()
        predictor.train_model(n_samples=600)

        test_locations = [
            (37.570, 126.982),  # 종각역
            (37.5814, 126.9880),  # 대동세무고등학교
            (37.5665, 126.9780),  # 서울역
        ]

        print("\n=== 보행자 위험도 예측 결과 ===")
        for i, (lat, lon) in enumerate(test_locations, 1):
            result = predictor.predict_risk(lat, lon)
            print(f"\n위치 {i}: 위도 {lat}, 경도 {lon}")
            print(f"위험도 점수: {result['risk_score']}/10")
            print(f"위험 수준: {result['risk_level']}")
            print(f"주변 총 사고 건수: {result['nearby_accidents']}건")
            print(f"보행자 관련 사고: {result['pedestrian_accidents']}건")

        print("\n=== 직접 좌표 입력 ===")
        print("종료하려면 Ctrl+C를 누르세요.")

        example_coords = [
            (37.570, 126.982, "종각역"),
            (37.5814, 126.9880, "대동세무고등학교"),
            (37.5665, 126.9780, "서울역"),
            (37.4979, 127.0276, "강남역")
        ]

        print("\n=== 추가 테스트 위치 ===")
        for lat, lon, name in example_coords:
            result = predictor.predict_risk(lat, lon)
            print(f"\n📍 {name} ({lat}, {lon}):")
            print(f"🔢 위험도 점수: {result['risk_score']}/10")
            print(f"⚠️  위험 수준: {result['risk_level']}")
            print(f"📊 주변 사고 분석:")
            print(f"   • 총 사고 건수: {result['nearby_accidents']}건")
            print(f"   • 보행자 관련 사고: {result['pedestrian_accidents']}건")
            print(f"   • 치명적 사고: {result['details'].get('fatal_accidents', 0)}건")
            print(f"   • 야간 사고: {result['details'].get('night_accidents', 0)}건")
            print(f"   • 주말 사고: {result['details'].get('weekend_accidents', 0)}건")

        print("\n시스템이 성공적으로 실행되었습니다!")

        print("\n=== 사각형 지역 위험도 분석 ===")
        print("A, B 두 지점의 좌표를 입력하여 사각형 지역의 위험도를 분석합니다.")

        try:
            print("\n📍 좌표 입력 (형식: 위도,경도):")
            a_input = input("A 지점 좌표: ")
            lat_a, lon_a = map(float, a_input.split(','))

            b_input = input("B 지점 좌표: ")
            lat_b, lon_b = map(float, b_input.split(','))

            print(f"\n분석 설정:")
            print(f"A 지점: ({lat_a}, {lon_a})")
            print(f"B 지점: ({lat_b}, {lon_b})")
            print("75m 간격으로 위험도 분석을 시작합니다...")

            danger_centers = predictor.analyze_rectangular_area(lat_a, lon_a, lat_b, lon_b)

            predictor.save_danger_centers_json(danger_centers)

            print(f"\n📊 첫 5개 격자점 정보:")
            for i, center in enumerate(danger_centers[:5], 1):
                print(f"{i}. 위도: {center['lat']}, 경도: {center['lon']}, 위험도: {center['weight']}/10")

            print(f"\n✅ 총 {len(danger_centers)}개 격자점 분석 완료")
            print("📄 결과가 danger_centers_output.json 파일로 저장되었습니다.")

        except ValueError:
            print("❌ 올바른 숫자 형식의 좌표를 입력해주세요.")
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")

    except Exception as e:
        print(f"시스템 초기화 중 오류 발생: {e}")
        print("CSV 파일 경로와 데이터 형식을 확인해주세요.")

if __name__ == "__main__":
    main()
