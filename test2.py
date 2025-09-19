import osmnx as ox
import networkx as nx
from geopy.distance import geodesic
import folium
import json
import os
import requests
import webbrowser
from dotenv import load_dotenv
from scipy.spatial import cKDTree
import numpy as np

load_dotenv()  # env 파일에서 Kakao API key 불러오기

def kakao_geocode(address, api_key):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
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

start = "종각역"
end = "대동세무고"
shownode = False

api_key = os.getenv('Kakao_REST_API_key')
start_coords = kakao_geocode(start, api_key)
end_coords = kakao_geocode(end, api_key)
print(f"출발지 {start} 좌표:", start_coords)
print(f"도착지 {end} 좌표:", end_coords)

# 도보 네트워크 그래프 불러오기 (출발지 기준 3km 반경)
G = ox.graph_from_point(start_coords, dist=3000, network_type='walk')

# 위험지역 JSON 로드
file_path = 'data/danger_centers_output.json'
with open(file_path, 'r', encoding='utf-8') as file:
    danger_centers_with_weight = json.load(file)

radius_m = 30  # 위험지역 반경(m)

# 1. 노드 좌표 배열, KD-tree 생성
node_coords = np.array([(data['y'], data['x']) for node_id, data in G.nodes(data=True)])
node_ids = list(G.nodes())
kdtree = cKDTree(node_coords)

danger_nodes = {}

# 2. KD-tree로 위험지역 주변 노드 후보군 선별, 정확 거리 계산 후 가중치 할당
approx_radius_deg = 0.0005  # 약 50m 반경(위도/경도 단위 근사)

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

# 3. 간선별로 가중치 계산 (노드 가중치 반영)
for u, v, k, data in G.edges(keys=True, data=True):
    base_length = data.get("length", 1.0)
    u_weight = danger_nodes.get(u, 1.0)
    v_weight = danger_nodes.get(v, 1.0)
    risk_factor = max(u_weight, v_weight)
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

# 8. 위험지역 원형 마커 추가
for center in danger_centers_with_weight:
    location = (center['lat'], center['lon'])
    weight = center['weight']
    if weight > 1:
        folium.Circle(
            location, radius=radius_m, color='red', fill=True, fill_opacity=0.3
        ).add_to(m)
        folium.map.Marker(
            location=location,
            icon=folium.DivIcon(
                html=f'<div style="font-size:10pt; font-weight:bold; color:black; text-align:center;">{weight:.1f}</div>'
            )
        ).add_to(m)
        folium.Circle(
            location, radius=radius_m, color='red', fill=True, fill_opacity=0.3,
            tooltip=f'위험 가중치: {weight}'
        ).add_to(m)

# 9. 출발/도착점 마커 표시
folium.Marker(route_latlng[0], popup=start, icon=folium.Icon(color='green')).add_to(m)
folium.Marker(route_latlng[-1], popup=end, icon=folium.Icon(color='red')).add_to(m)

# 10. 지도 HTML 저장 및 웹 브라우저 열기
out_dir = "data"
os.makedirs(out_dir, exist_ok=True)
file_path = os.path.join(out_dir, "route_map.html")
m.save(file_path)
print(f"저장된 지도 파일: {file_path}")

webbrowser.open(f"file://{os.path.abspath(file_path)}")

# 11. 경로 정보 JSON 저장 (선택)
route_json = {
    "nodes": {n: {"lat": G.nodes[n]['y'], "lon": G.nodes[n]['x']} for n in route},
    "coords": route_latlng
}
with open(os.path.join(out_dir, "route_coords.json"), "w", encoding="utf-8") as f:
    json.dump(route_json, f, indent=4, ensure_ascii=False)
print(f"저장된 노드 경로 파일: {os.path.join(out_dir, 'route_coords.json')}")

