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

# 지도 시각화를 위한 라이브러리
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic
import folium
import requests
import webbrowser
from dotenv import load_dotenv
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    import locale
    try:
        os.system('chcp 65001 > nul')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

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

            # 점수를 1-30 범위로 확장하여 차이를 극대화
            total_weight = (
                features['total_accidents'] * 0.05 +
                features['pedestrian_accidents'] * 0.4 +
                features['fatal_accidents'] * 1.0 +
                features['night_accidents'] * 0.1 +
                features['weekend_accidents'] * 0.05
            )

            # 심각도 가중치 증가
            severity_factor = features['severity_weighted_score'] * 0.02

            # 기본 점수를 1점으로 설정하고, 사고가 있을 때만 점수 증가
            if total_weight > 0 or severity_factor > 0:
                # 지수 함수 사용으로 점수 차이를 극대화 (제한 없음)
                raw_score = (total_weight + severity_factor) ** 1.5 * 3.0
                risk_score = 1.0 + raw_score
            else:
                risk_score = 1.0

            # 최대값 제한 제거 - 최소값만 1.0으로 제한
            risk_score = max(1.0, risk_score)

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

            # 최대값 제한 제거 - 최소값만 1.0으로 제한
            risk_score = max(1, risk_score)

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
        if score <= 5:
            return "매우 낮음"
        elif score <= 20:
            return "낮음"
        elif score <= 50:
            return "보통"
        elif score <= 100:
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

# 지도 시각화 클래스
class RouteVisualizer:
    def __init__(self):
        self.api_key = os.getenv('Kakao_REST_API_key')

    def interpolate_coordinates_1m(self, route_coords):
        """경로 좌표를 1m 간격으로 보간하는 함수"""
        interpolated_coords = []

        for i in range(len(route_coords) - 1):
            start_lat, start_lon = route_coords[i]
            end_lat, end_lon = route_coords[i + 1]

            # 두 점 사이의 거리 계산 (미터)
            distance = geodesic((start_lat, start_lon), (end_lat, end_lon)).meters

            # 1m 간격으로 나누기 위한 점의 개수 계산
            num_points = max(1, int(distance))

            # 시작점 추가
            interpolated_coords.append({
                "latitude": round(start_lat, 8),
                "longitude": round(start_lon, 8)
            })

            # 보간된 점들 추가
            for j in range(1, num_points):
                ratio = j / num_points

                # 선형 보간으로 중간 좌표 계산
                lat = start_lat + (end_lat - start_lat) * ratio
                lon = start_lon + (end_lon - start_lon) * ratio

                interpolated_coords.append({
                    "latitude": round(lat, 8),
                    "longitude": round(lon, 8)
                })

        # 마지막 점 추가
        if route_coords:
            last_lat, last_lon = route_coords[-1]
            interpolated_coords.append({
                "latitude": round(last_lat, 8),
                "longitude": round(last_lon, 8)
            })

        return interpolated_coords

    def kakao_geocode(self, address):
        url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {"Authorization": f"KakaoAK {self.api_key}"}
        params = {"query": address}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return None
        result = response.json()
        if 'documents' in result and len(result['documents']) > 0:
            x = float(result['documents'][0]['x'])
            y = float(result['documents'][0]['y'])
            return (y, x)  # 위도, 경도 순서
        else:
            print("No matching address found.")
            return None

    def visualize_original_route(self, start_location, end_location, shownode=False):
        """위험도 가중치를 적용하지 않은 원래 최단 경로 시각화"""
        # 주소를 좌표로 변환
        start_coords = self.kakao_geocode(start_location)
        end_coords = self.kakao_geocode(end_location)

        if not start_coords or not end_coords:
            print("주소를 좌표로 변환할 수 없습니다.")
            return

        print(f"출발지 {start_location} 좌표:", start_coords)
        print(f"도착지 {end_location} 좌표:", end_coords)

        # 도보 네트워크 그래프 불러오기 (출발지 기준 3km 반경)
        G = ox.graph_from_point(start_coords, dist=3000, network_type='walk')

        # 출발 / 도착 노드 탐색
        start_node = ox.distance.nearest_nodes(G, X=start_coords[1], Y=start_coords[0])
        end_node = ox.distance.nearest_nodes(G, X=end_coords[1], Y=end_coords[0])

        # 일반적인 최단 경로 탐색 (거리만 고려)
        route = nx.shortest_path(G, start_node, end_node, weight='length')

        # 경로 노드 좌표 추출
        route_latlng = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]

        # folium 지도 생성 및 경로 표시
        m = folium.Map(location=route_latlng[0], zoom_start=15)
        folium.PolyLine(route_latlng, color='red', weight=5, opacity=0.8).add_to(m)

        if shownode == True:
            # 모든 기본 노드 검은 점으로 지도에 추가
            for node_id, data in G.nodes(data=True):
                folium.CircleMarker(
                    location=(data['y'], data['x']),
                    radius=2,
                    color='black',
                    fill=True,
                    fill_opacity=1
                ).add_to(m)

        # 출발/도착점 마커 표시
        folium.Marker(route_latlng[0], popup=f"{start_location} (출발)", icon=folium.Icon(color='green')).add_to(m)
        folium.Marker(route_latlng[-1], popup=f"{end_location} (도착)", icon=folium.Icon(color='red')).add_to(m)

        # 지도 HTML 저장
        out_dir = "data"
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, "original_route_map.html")
        m.save(file_path)
        print(f"원래 경로 지도 파일 저장: {file_path}")

    def visualize_route(self, start_location, end_location, danger_centers_file="danger_centers_output.json", shownode=False):
        # 주소를 좌표로 변환
        start_coords = self.kakao_geocode(start_location)
        end_coords = self.kakao_geocode(end_location)

        if not start_coords or not end_coords:
            print("주소를 좌표로 변환할 수 없습니다.")
            return

        print(f"출발지 {start_location} 좌표:", start_coords)
        print(f"도착지 {end_location} 좌표:", end_coords)

        # 도보 네트워크 그래프 불러오기 (출발지 기준 3km 반경)
        G = ox.graph_from_point(start_coords, dist=3000, network_type='walk')

        # 위험지역 JSON 로드
        if os.path.exists(danger_centers_file):
            with open(danger_centers_file, 'r', encoding='utf-8') as file:
                danger_centers_with_weight = json.load(file)
        else:
            print(f"위험지역 파일 {danger_centers_file}을 찾을 수 없습니다.")
            return

        radius_m = 30  # 위험지역 반경(m)

        # 1. 노드 좌표 배열, KD-tree 생성
        node_coords = np.array([(data['y'], data['x']) for node_id, data in G.nodes(data=True)])
        node_ids = list(G.nodes())
        kdtree = cKDTree(node_coords)

        danger_nodes = {}

        # 2. KD-tree로 위험지역 주변 노드 후보군 선별, 정확 거리 계산 후 가중치 할당
        approx_radius_deg = 0.00025  # 약 50m 반경(위도/경도 단위 근사)

        for danger_area in danger_centers_with_weight:
            center = (danger_area["lat"], danger_area["lon"])
            weight = danger_area["weight"]

            center_coord = np.array([center[0], center[1]])
            candidate_idxs = kdtree.query_ball_point(center_coord, approx_radius_deg)

            for idx in candidate_idxs:
                node_coord = node_coords[idx]
                dist = geodesic(center, node_coord).meters
                if dist <= radius_m:
                    node_id = node_ids[idx]
                    if node_id not in danger_nodes or danger_nodes[node_id] < weight:
                        danger_nodes[node_id] = weight

        # 3. 간선별로 가중치 계산 (거리 효율성 우선, 적당한 위험도 회피)
        for u, v, k, data in G.edges(keys=True, data=True):
            base_length = data.get("length", 1.0)
            u_weight = danger_nodes.get(u, 1.0)
            v_weight = danger_nodes.get(v, 1.0)
            max_risk = max(u_weight, v_weight)

            # 거리 효율성을 최우선으로 하고 위험도는 최소한만 고려
            if max_risk >= 500:
                risk_factor = 1.0 + (max_risk * 0.01)  # 500점 이상: 아주 약간 회피
            elif max_risk >= 200:
                risk_factor = 1.0 + (max_risk * 0.005)  # 200-500점: 거의 무시
            elif max_risk >= 100:
                risk_factor = 1.0 + (max_risk * 0.002)  # 100-200점: 미미하게 고려
            else:
                risk_factor = 1.0  # 100점 미만: 거리만 고려

            data["custom_weight"] = base_length * risk_factor

        # 4. 출발 / 도착 노드 탐색
        start_node = ox.distance.nearest_nodes(G, X=start_coords[1], Y=start_coords[0])
        end_node = ox.distance.nearest_nodes(G, X=end_coords[1], Y=end_coords[0])

        # 5. 가중치(custom_weight) 기준 최단 경로 탐색
        route = nx.shortest_path(G, start_node, end_node, weight='custom_weight')

        # 6. 경로 노드 좌표 추출
        route_latlng = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]

        # 7. folium 지도 생성 및 경로 표시
        m = folium.Map(location=route_latlng[0], zoom_start=15)
        folium.PolyLine(route_latlng, color='blue', weight=5, opacity=0.8).add_to(m)

        if shownode == True:
            # 7-1. 모든 기본 노드 검은 점으로 지도에 추가
            for node_id, data in G.nodes(data=True):
                folium.CircleMarker(
                    location=(data['y'], data['x']),
                    radius=2,
                    color='black',
                    fill=True,
                    fill_opacity=1
                ).add_to(m)

        # 8. 위험지역 원형 마커 추가 (점수에 따라 자연스럽게 색상 변경)
        for center in danger_centers_with_weight:
            location = (center['lat'], center['lon'])
            weight = center['weight']
            if weight > 1:
                # 점수에 따라 자연스럽게 색상과 투명도 계산
                # 1점(연한 노랑)에서 1000점(진한 빨강)까지 자연스럽게 변화

                # 점수를 0-1 범위로 정규화 (로그 스케일 적용)
                import math
                max_score = 1000  # 최대 예상 점수
                normalized = min(1.0, math.log(weight + 1) / math.log(max_score + 1))

                # HSV 색상 공간에서 색조(Hue) 변경: 60도(노랑)에서 0도(빨강)으로
                hue = 60 * (1 - normalized)  # 60(노랑) → 0(빨강)

                # HSV를 RGB로 변환
                import colorsys
                rgb = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)  # 채도 0.8, 명도 0.9

                # RGB를 16진수 색상코드로 변환
                color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

                # 투명도도 점수에 따라 자연스럽게 변화 (전체적으로 낮게)
                opacity = 0.1 + (normalized * 0.3)  # 0.2~0.6 범위로 낮춤

                folium.Circle(
                    location, radius=radius_m, color=color, fill=True,
                    fill_color=color, fill_opacity=opacity
                ).add_to(m)

                # 텍스트 색상도 배경에 따라 조정
                text_color = 'white' if normalized > 0.5 else 'black'

                folium.map.Marker(
                    location=location,
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:8pt; font-weight:bold; color:{text_color}; text-align:center; background-color:{color}; border-radius:50%; width:20px; height:20px; line-height:20px; opacity:0.8;">{weight:.0f}</div>'
                    )
                ).add_to(m)
                folium.Circle(
                    location, radius=radius_m, color=color, fill=True,
                    fill_color=color, fill_opacity=opacity,
                    tooltip=f'위험도: {weight:.1f}점'
                ).add_to(m)

        # 9. 출발/도착점 마커 표시
        folium.Marker(route_latlng[0], popup=start_location, icon=folium.Icon(color='green')).add_to(m)
        folium.Marker(route_latlng[-1], popup=end_location, icon=folium.Icon(color='red')).add_to(m)

        # 10. 지도 HTML 저장 및 웹 브라우저 열기
        out_dir = "data"
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, "route_map.html")
        m.save(file_path)
        print(f"저장된 지도 파일: {file_path}")

        webbrowser.open(f"file://{os.path.abspath(file_path)}")

        # 11. 1m 간격 좌표 생성 및 JSON 저장
        print("1m 간격으로 경로 좌표를 생성하고 있습니다...")
        interpolated_coords = self.interpolate_coordinates_1m(route_latlng)

        # 1m 간격 좌표 JSON 저장
        detailed_route_json = {
            "route_info": {
                "route_type": "safety_optimized_path",
                "start_location": start_location,
                "end_location": end_location,
                "total_points": len(interpolated_coords),
                "approximate_distance_meters": len(interpolated_coords)
            },
            "coordinates": interpolated_coords
        }

        detailed_file_path = os.path.join(out_dir, "detailed_route_1m.json")
        with open(detailed_file_path, "w", encoding="utf-8") as f:
            json.dump(detailed_route_json, f, indent=4, ensure_ascii=False)
        print(f"1m 간격 상세 경로 파일 저장: {detailed_file_path}")

        # 12. 기존 경로 정보 JSON 저장 (선택)
        route_json = {
            "route_type": "safety_optimized_path",
            "nodes": {n: {"lat": G.nodes[n]['y'], "lon": G.nodes[n]['x']} for n in route},
            "coords": route_latlng
        }
        with open(os.path.join(out_dir, "route_coords.json"), "w", encoding="utf-8") as f:
            json.dump(route_json, f, indent=4, ensure_ascii=False)
        print(f"저장된 노드 경로 파일: {os.path.join(out_dir, 'route_coords.json')}")

def main():
    try:
        print("=== 통합 보행자 위험도 분석 및 경로 시각화 시스템 ===")

        # 1. 위험도 예측 모델 초기화 및 훈련
        print("\n1. 보행자 위험도 예측 시스템 초기화...")
        predictor = PedestrianRiskPredictor()
        predictor.train_model(n_samples=600)

        # 2. 사용자로부터 경로 출발지/도착지 입력받기
        print("\n=== 경로 설정 ===")
        start_location = input("출발지를 입력하세요 (예: 종각역): ").strip()
        end_location = input("도착지를 입력하세요 (예: 대동세무고): ").strip()

        if not start_location:
            start_location = "종각역"
            print("출발지가 입력되지 않아 기본값 '종각역'을 사용합니다.")
        if not end_location:
            end_location = "대동세무고"
            print("도착지가 입력되지 않아 기본값 '대동세무고'를 사용합니다.")

        # 3. 출발지/도착지 좌표로 분석 지역 자동 설정
        visualizer = RouteVisualizer()

        if not visualizer.api_key:
            print("Kakao API 키가 설정되지 않았습니다. .env 파일에 Kakao_REST_API_key를 설정해주세요.")
            return

        print(f"\n출발지와 도착지 좌표를 확인하고 있습니다...")
        start_coords = visualizer.kakao_geocode(start_location)
        end_coords = visualizer.kakao_geocode(end_location)

        if not start_coords or not end_coords:
            print("주소를 좌표로 변환할 수 없습니다. 기본 좌표를 사용합니다.")
            lat_a, lon_a = 37.570, 126.982
            lat_b, lon_b = 37.5814, 126.9880
        else:
            lat_a, lon_a = start_coords
            lat_b, lon_b = end_coords
            print(f"출발지 {start_location} 좌표: ({lat_a}, {lon_a})")
            print(f"도착지 {end_location} 좌표: ({lat_b}, {lon_b})")

        # 4. 사각형 지역 위험도 분석
        print("\n=== 사각형 지역 위험도 분석 ===")
        print("출발지와 도착지를 기준으로 분석 지역이 자동 설정되었습니다.")
        print("75m 간격으로 위험도 분석을 시작합니다...")

        danger_centers = predictor.analyze_rectangular_area(lat_a, lon_a, lat_b, lon_b)
        predictor.save_danger_centers_json(danger_centers)

        print(f"\n📊 첫 5개 격자점 정보:")
        for i, center in enumerate(danger_centers[:5], 1):
            print(f"{i}. 위도: {center['lat']}, 경도: {center['lon']}, 위험도: {center['weight']}/10")

        # 5. 경로 시각화 (두 가지 버전)
        print("\n=== 경로 시각화 ===")

        # 5-1. 원래 최단 경로 (위험도 미고려)
        print(f"\n1. 원래 최단 경로 지도를 생성합니다... ({start_location} → {end_location})")
        visualizer.visualize_original_route(start_location, end_location, shownode=False)

        # 5-2. 안전 최적화 경로 (위험도 고려)
        print(f"\n2. 안전 최적화 경로 지도를 생성합니다... ({start_location} → {end_location})")
        visualizer.visualize_route(start_location, end_location, shownode=False)

        print("\n두 가지 경로 지도가 모두 생성되었습니다!")

        print("\n✅ 통합 분석이 완료되었습니다!")
        print("📄 생성된 파일:")
        print("- danger_centers_output.json: 위험도 격자 데이터")
        print("\n📍 원래 최단 경로 (위험도 미고려):")
        print("- data/original_route_map.html: 원래 최단 경로 지도")
        print("\n🛡️ 안전 최적화 경로 (위험도 고려):")
        print("- data/route_map.html: 안전 최적화 경로 지도")
        print("- data/route_coords.json: 안전 경로 좌표 데이터")
        print("- data/detailed_route_1m.json: 안전 경로 1m 간격 좌표")

    except Exception as e:
        print(f"시스템 실행 중 오류 발생: {e}")
        print("CSV 파일 경로와 데이터 형식, API 키 설정을 확인해주세요.")

if __name__ == "__main__":
    main()