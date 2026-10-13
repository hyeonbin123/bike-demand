// [파일 역할] 부족 예측 대시보드의 동작 전부. API만 읽어(/health, /stations, /shortage-risk, /stations/{id})
// 지도 원, 대여소 표, 대여소별 6시간 막대를 그린다. 빌드 없이 index.html이 defer로 불러온다.
// [설계] 즉시 실행 함수로 감싸 전역 변수를 만들지 않는다. 화면에 넣는 API 문자열은 모두 textContent로만 넣어
// HTML로 해석되지 않게 한다(대여소 이름 등이 스크립트로 실행될 여지를 없앰, 커밋 8ec2fa3).
// [흐름] 처음 열 때 initMap → refresh. refresh가 상태·자치구·부족 목록을 동시에 불러오고, 대여소를 고르면
// selectStation이 상세를 불러온다. 조건을 빨리 바꿔도 이전 요청은 AbortController로 취소해 늦게 온 옛 응답이
// 화면을 덮지 않게 한다.

(() => {
  "use strict";

  // 자주 쓰는 화면 요소를 id로 한 번만 찾아 ui 객체에 모아 둔다.
  const byId = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    "filters", "hours", "district", "refresh", "district-message", "snapshot-time",
    "health-message", "risk-note", "risk-status", "map-count", "map-message",
    "station-count", "risk-rows", "empty-message", "detail", "detail-title",
    "detail-snapshot", "detail-status", "forecast-bars", "detail-note",
    "detail-version", "risk-version",
  ].map((id) => [id, byId(id)]));
  // 시각 표시는 브라우저 시간대와 상관없이 Asia/Seoul로 고정한다(화면 하단 '모든 시각은 KST'). 24시간제.
  const stampFormat = new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23",
  });
  const hourFormat = new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  });
  // 화면 상태: Leaflet 지도와 원 묶음 층, 진행 중인 목록 조회·상세 조회의 취소 장치, 대여소ID로 지도 원과 표
  // 행을 찾는 표(선택 강조에 씀).
  let map = null;
  let circles = null;
  let queryController = null;
  let detailController = null;
  const markers = new Map();
  const rowButtons = new Map();

  // 요소를 만들고 글을 textContent로 넣는 도우미. innerHTML을 쓰지 않는 것이 이 파일의 보안 규칙이다.
  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // ISO 시각 문자열을 'MM. DD. HH:MM:SS' 형태 KST로. 값이 없으면 '없음'.
  function stamp(value) {
    return value ? stampFormat.format(new Date(value)) : "없음";
  }

  // 부족 대수 구간: 10대 이상 high, 5대 이상 medium, 그 밖 low. 범례·CSS 클래스·원 색이 이 구간을 함께 쓴다.
  function tier(value) {
    return value >= 10 ? "high" : value >= 5 ? "medium" : "low";
  }

  // 부족 대수 표시 문자열.
  function shortfallText(value) {
    // API는 양수만 반환하지만 0.05 미만은 한 자리 반올림 뒤 0.0이 된다.
    return value === 0 ? "0.1 미만" : value.toFixed(1);
  }

  // 좌표가 있고 위경도 범위 안인지. 좌표가 없는 대여소(대여소 정보에 없는 곳)는 표에는 남기고 지도에서만
  // 뺀다.
  function hasCoordinates(station) {
    return Number.isFinite(station.lat) && Number.isFinite(station.lon)
      && Math.abs(station.lat) <= 90 && Math.abs(station.lon) <= 180;
  }

  // fetch 도우미. 호출한 쪽의 취소 신호와 15초 제한 시간을 함께 걸고, 실패 응답이면 상태 코드와 API의
  // detail을 담은 Error를 던진다.
  // 자체 AbortController를 따로 두는 이유: 바깥 취소(조건 변경)와 제한 시간 초과를 같은 신호로 끊되, 어느 쪽
  // 때문인지 catch에서 구분하려는 것이다.
  async function request(url, signal) {
    const controller = new AbortController();
    const cancel = () => controller.abort();
    signal.addEventListener("abort", cancel, { once: true });
    if (signal.aborted) controller.abort();
    const timer = setTimeout(cancel, 15000);
    try {
      const response = await fetch(url, { signal: controller.signal, headers: { Accept: "application/json" } });
      // API 오류는 docs/api.md 형식 {detail: ...}이라 본문의 detail을 읽어 둔다. 본문이 JSON이 아니어도 빈
      // 객체로 넘어간다.
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        const error = new Error("request failed");
        error.status = response.status;
        error.detail = body.detail;
        throw error;
      }
      return await response.json();
    } catch (error) {
      // 우리 쪽 신호만 취소됐으면 제한 시간 초과다. 바깥에서 취소한 경우는 원래 오류를 그대로 던지고, 호출한
      // 쪽이 signal.aborted로 조용히 끝낸다.
      if (controller.signal.aborted && !signal.aborted) throw new Error("timeout");
      throw error;
    } finally {
      clearTimeout(timer);
      signal.removeEventListener("abort", cancel);
    }
  }

  // 오류를 사용자 문구로 바꾼다. 503의 detail 값(no predictions, no recent snapshot, database unavailable)은
  // docs/api.md에 정한 것이라 각각 다른 안내를 보인다.
  function errorMessage(error) {
    if (error.status === 503) {
      if (error.detail === "no predictions") return "예측 데이터가 아직 없습니다. 예측이 준비된 뒤 새로고침해 주세요.";
      if (error.detail === "no recent snapshot") return "최근 30분 이내의 자전거 정보가 없습니다. 수집이 재개된 뒤 새로고침해 주세요.";
      return "서비스 데이터를 지금 조회할 수 없습니다. 잠시 후 새로고침해 주세요.";
    }
    if (error.status === 404) return "이 대여소를 찾을 수 없습니다. 목록을 새로고침해 주세요.";
    return "정보를 불러오지 못했습니다. 연결을 확인한 뒤 새로고침해 주세요.";
  }

  // Leaflet 지도를 서울시청 좌표 중심, 확대 11로 연다. Leaflet을 불러오지 못했거나(인터넷 끊김 등) 만들다
  // 실패하면 지도 없이 표만 쓰도록 안내만 남긴다. 페이지를 스크롤하다 지도가 확대되지 않게 휠 확대는 끈다.
  function initMap() {
    if (!window.L) {
      ui["map-message"].textContent = "지도 도구를 불러오지 못했습니다. 대여소 목록은 계속 사용할 수 있습니다.";
      return;
    }
    try {
      map = window.L.map("map", { scrollWheelZoom: false }).setView([37.5665, 126.9780], 11);
      // OpenStreetMap 배경 타일. 사용 조건에 따라 출처(attribution)를 지도에 표시한다. 타일을 못 받으면 안내
      // 문구만 띄운다.
      window.L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).on("tileerror", () => {
        ui["map-message"].textContent = "배경 지도를 불러오지 못했습니다. 부족 예측은 원과 목록으로 확인해 주세요.";
      }).addTo(map);
      circles = window.L.layerGroup().addTo(map);
    } catch (_) {
      map = null;
      circles = null;
      ui["map-message"].textContent = "지도를 표시할 수 없습니다. 대여소 목록을 이용해 주세요.";
    }
  }

  // 상세 영역을 처음 안내 상태로 되돌리고, 진행 중인 상세 요청은 취소한다.
  function resetDetail() {
    detailController?.abort();
    ui.detail.setAttribute("aria-busy", "false");
    ui["detail-title"].textContent = "대여소를 선택하세요";
    ui["detail-snapshot"].textContent = "";
    ui["detail-status"].textContent = "지도 위 원이나 목록의 대여소를 선택하면 앞으로 6시간의 예측을 보여 드립니다.";
    ui["forecast-bars"].replaceChildren();
    ui["detail-note"].hidden = true;
    ui["detail-version"].textContent = "";
  }

  // 새 조회 전에 지도 원·표·개수·버전 표시와 선택 상태를 비운다.
  function clearResults() {
    circles?.clearLayers();
    markers.clear();
    rowButtons.clear();
    ui["risk-rows"].replaceChildren();
    ui["station-count"].textContent = "—";
    ui["map-count"].textContent = "—";
    ui["risk-version"].textContent = "";
    ui["empty-message"].hidden = false;
    ui["empty-message"].textContent = "데이터를 불러오고 있습니다.";
    resetDetail();
  }

  // /health에서 전체 대여소 중 가장 최근 수집 시각을 보여 준다. 30분보다 오래됐으면 경고한다(API의 스냅샷
  // 나이 기본값 30분과 같은 기준).
  async function loadHealth(signal) {
    ui["snapshot-time"].textContent = "확인 중…";
    ui["health-message"].textContent = "";
    try {
      const data = await request("/health", signal);
      if (signal.aborted) return;
      ui["snapshot-time"].textContent = stamp(data.latest_snapshot_at);
      const age = Date.now() - new Date(data.latest_snapshot_at).getTime();
      ui["health-message"].textContent = !data.latest_snapshot_at ? "아직 수집된 스냅샷이 없습니다."
        : age > 30 * 60 * 1000 ? "30분 이상 지난 정보입니다."
        : "전체 대여소 중 가장 최근 수집 시각";
    } catch (error) {
      if (signal.aborted) return;
      ui["snapshot-time"].textContent = "확인할 수 없음";
      ui["health-message"].textContent = errorMessage(error);
    }
  }

  // /stations로 자치구 선택지를 다시 만든다. 지금 고른 자치구는 목록에 없더라도 남겨 선택이 풀리지 않게 하고,
  // 한국어 순서로 정렬한다. 실패하면 현재 조건 그대로 조회하고 안내만 띄운다.
  async function loadDistricts(signal) {
    try {
      const stations = await request("/stations", signal);
      if (signal.aborted) return;
      const selected = ui.district.value;
      const districts = new Set(stations.map((station) => station.district).filter(Boolean));
      if (selected) districts.add(selected);
      const all = element("option", "", "서울 전체");
      all.value = "";
      const options = [...districts].sort((a, b) => a.localeCompare(b, "ko")).map((district) => {
        const option = element("option", "", district);
        option.value = district;
        return option;
      });
      ui.district.replaceChildren(all, ...options);
      ui.district.value = selected;
      ui.district.disabled = false;
      ui["district-message"].textContent = "";
    } catch (_) {
      if (!signal.aborted) ui["district-message"].textContent = "자치구 목록을 갱신하지 못했습니다. 현재 조건으로 조회하며 새로고침 시 다시 시도합니다.";
    }
  }

  // 부족 목록을 표 행과 지도 원으로 그린다. 좌표가 없는 대여소는 표에만 넣고, 지도가 없으면 원은 건너뛴다.
  // 끝에 좌표가 있는 원들이 모두 보이게 지도 범위를 맞춘다.
  function renderStations(stations) {
    const coordinates = [];
    for (const station of stations) {
      const row = element("tr");
      const nameCell = element("td");
      const button = element("button", "station-button", station.station_name || station.station_id);
      button.type = "button";
      button.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-controls", "detail");
      button.addEventListener("click", () => selectStation(station));
      rowButtons.set(station.station_id, { row, button });
      nameCell.append(button, element("span", "station-meta",
        `${station.district || "자치구 미상"} · ${station.station_id}${hasCoordinates(station) ? "" : " · 좌표 없음"}`));
      const shortage = element("td");
      shortage.append(element("span", `shortfall ${tier(station.shortfall)}`, shortfallText(station.shortfall)));
      row.append(nameCell, element("td", "", String(station.bike_count)),
        element("td", "", station.expected_rentals.toFixed(1)), shortage);
      ui["risk-rows"].append(row);
      if (!hasCoordinates(station)) continue;
      coordinates.push([station.lat, station.lon]);
      if (!circles) continue;
      // 원 색은 dashboard.css의 low·medium·high 변수와 같은 값. 반지름은 6 + 2·√부족대수(최대 23)라 부족
      // 대수가 커져도 반지름은 제곱근으로 천천히 커지고 23px에서 멈춘다.
      const color = { low: "#926820", medium: "#bc4d20", high: "#a32936" }[tier(station.shortfall)];
      const marker = window.L.circleMarker([station.lat, station.lon], {
        radius: Math.min(23, 6 + Math.sqrt(station.shortfall) * 2),
        color, fillColor: color, fillOpacity: 0.65, weight: 2,
      }).addTo(circles);
      // Leaflet은 문자열을 HTML로 해석한다. API 값은 textContent인 DOM으로만 전달한다.
      marker.bindTooltip(element("span", "", `${station.station_name || station.station_id} · 부족 ${shortfallText(station.shortfall)}대`));
      marker.on("click", () => selectStation(station));
      markers.set(station.station_id, marker);
    }
    ui["station-count"].textContent = `${stations.length.toLocaleString("ko-KR")}곳`;
    ui["map-count"].textContent = `좌표 ${coordinates.length}곳 · 미상 ${stations.length - coordinates.length}곳`;
    if (map && coordinates.length) map.fitBounds(coordinates, { padding: [30, 30], maxZoom: 14 });
  }

  // /shortage-risk를 API 최대 limit(500)으로 불러 그린다. 결과 줄에 자치구·시간·곳 수를 적고, 500곳이면 조회
  // 상한에 닿았다고 알린다. 빈 목록은 '자전거가 충분하다'는 뜻이 아니라는 점(최신 스냅샷·예측이 없는 대여소는
  // 빠짐)을 함께 적는다.
  async function loadRisk(signal, hours, district) {
    const params = new URLSearchParams({ hours, limit: "500" });
    if (district) params.set("district", district);
    try {
      const data = await request(`/shortage-risk?${params}`, signal);
      if (signal.aborted) return;
      renderStations(data.stations);
      ui["risk-note"].textContent = data.note || "반납은 반영하지 않은 값";
      const count = data.stations.length;
      ui["risk-status"].textContent = `${district || "서울 전체"} · 앞으로 ${data.hours}시간 · ${count === 500 ? "부족 대수가 큰 500곳 표시 (조회 상한)" : `부족 예상 ${count}곳`}`;
      ui["empty-message"].hidden = count > 0;
      ui["empty-message"].textContent = "이 조건에서 표시할 부족 예상 대여소가 없습니다. 최신 스냅샷이나 필요한 시간의 예측이 없는 대여소는 제외되므로, 모든 대여소에 자전거가 충분하다는 뜻은 아닙니다.";
      ui["risk-version"].textContent = `조회 시각 ${stamp(data.generated_at)} KST · 예측 버전 ${data.model_version}`;
    } catch (error) {
      if (signal.aborted) return;
      // 이전 조건의 원·표를 성공 응답처럼 남겨 두지 않는다.
      circles?.clearLayers();
      ui["risk-rows"].replaceChildren();
      ui["risk-status"].classList.add("error");
      ui["risk-status"].textContent = errorMessage(error);
      ui["empty-message"].hidden = false;
      ui["empty-message"].textContent = "조회 결과가 없습니다. 위 안내를 확인한 뒤 새로고침해 주세요.";
    }
  }

  // 앞으로 6개 시간의 예측 막대. 시간은 epoch 밀리초 정시로 맞춰 서버 값과 짝짓는다. 막대 높이는 6시간 중
  // 최대값(최소 1) 대비 비율, 예측이 없는 시간은 빗금 칸과 '예측 없음'.
  function renderBars(predictions, requestedAt) {
    const values = new Map(predictions.map((item) => [new Date(item.hour_start).getTime(), item.predicted_rentals]));
    // 기본 시작은 요청한 순간이 든 정시. 아래에서 서버가 준 시간 범위로 보정한다.
    let first = Math.floor(requestedAt / 3600000) * 3600000;
    if (values.size) {
      // 요청 중 정시를 넘거나 기기 시계가 달라도 서버가 준 시간대를 잘라내지 않는다.
      first = Math.min(first, ...values.keys());
      first = Math.max(first, Math.max(...values.keys()) - 5 * 3600000);
    }
    const maximum = Math.max(1, ...predictions.map((item) => item.predicted_rentals));
    for (let i = 0; i < 6; i += 1) {
      const hour = first + i * 3600000;
      const value = values.get(hour);
      const label = hourFormat.format(new Date(hour));
      const valueText = value === undefined ? "예측 없음" : `${value.toFixed(2)}대`;
      const item = element("li");
      item.setAttribute("aria-label", `${label} ${valueText}`);
      const track = element("div", "bar-track");
      track.setAttribute("aria-hidden", "true");
      const fill = element("div", value === undefined ? "bar-missing" : "bar-fill");
      if (value !== undefined) fill.style.height = `${Math.max(0, value) / maximum * 100}%`;
      track.append(fill);
      item.append(element("span", "bar-value", valueText), track, element("span", "bar-hour", label));
      ui["forecast-bars"].append(item);
    }
  }

  // 대여소 하나를 고르면: 이전 상세 요청 취소 → 표 행·지도 원 강조(aria-pressed) → 상세 영역을 불러오는
  // 중으로 바꾸고 제목에 포커스(화면 낭독기가 바뀐 곳을 읽게) → /stations/{id}를 불러 스냅샷·막대·버전 표시.
  // 스냅샷이 30분보다 오래됐으면 함께 알린다.
  async function selectStation(station) {
    detailController?.abort();
    detailController = new AbortController();
    const { signal } = detailController;
    const requestedAt = Date.now();
    for (const [id, entry] of rowButtons) {
      entry.row.classList.toggle("selected", id === station.station_id);
      entry.button.setAttribute("aria-pressed", String(id === station.station_id));
    }
    markers.get(station.station_id)?.openTooltip();
    if (map && hasCoordinates(station)) map.panTo([station.lat, station.lon]);
    ui.detail.setAttribute("aria-busy", "true");
    ui["detail-title"].textContent = station.station_name || station.station_id;
    ui["detail-snapshot"].textContent = "";
    ui["detail-status"].textContent = "시간별 예측을 불러오고 있습니다.";
    ui["forecast-bars"].replaceChildren();
    ui["detail-note"].hidden = true;
    ui["detail-version"].textContent = "";
    ui["detail-title"].focus({ preventScroll: true });
    ui.detail.scrollIntoView({ block: "nearest" });
    try {
      // 대여소ID는 URL 경로에 들어가므로 encodeURIComponent로 감싼다.
      const data = await request(`/stations/${encodeURIComponent(station.station_id)}`, signal);
      if (signal.aborted) return;
      ui["detail-title"].textContent = data.station.station_name || data.station.station_id;
      ui["detail-snapshot"].textContent = data.snapshot
        ? `자전거 ${data.snapshot.bike_count}대 · ${stamp(data.snapshot.fetched_at)} KST 수집${requestedAt - new Date(data.snapshot.fetched_at).getTime() > 1800000 ? " · 30분 이상 지난 정보" : ""}`
        : "이 대여소의 스냅샷이 없습니다.";
      ui["detail-status"].textContent = data.predictions.length
        ? `현재 시간부터 6개 시간대 중 ${data.predictions.length}개 예측 제공 · 반납 미반영`
        : "선택된 예측 버전에 이 대여소의 앞으로 6시간 예측이 없습니다.";
      renderBars(data.predictions, requestedAt);
      ui["detail-note"].hidden = false;
      ui["detail-version"].textContent = `예측 버전 ${data.model_version || "없음"}`;
    } catch (error) {
      if (!signal.aborted) ui["detail-status"].textContent = errorMessage(error);
    } finally {
      if (!signal.aborted) ui.detail.setAttribute("aria-busy", "false");
    }
  }

  // 조회 한 번: 이전 조회 취소 → 결과 비우기 → 새로고침 버튼 비활성 → 상태·자치구·부족 목록을 동시에 요청. 세
  // 요청이 끝나고 이 조회가 취소되지 않았을 때만 버튼을 다시 켠다(더 새 조회가 버튼 상태를 맡음).
  async function refresh() {
    queryController?.abort();
    queryController = new AbortController();
    const { signal } = queryController;
    const hours = ui.hours.value;
    const district = ui.district.value;
    clearResults();
    ui["risk-status"].classList.remove("error");
    ui["risk-status"].textContent = "예측 정보를 불러오는 중입니다.";
    ui.refresh.disabled = true;
    await Promise.all([loadHealth(signal), loadDistricts(signal), loadRisk(signal, hours, district)]);
    if (!signal.aborted) ui.refresh.disabled = false;
  }

  // 폼 제출(새로고침 버튼)은 페이지 이동을 막고 다시 조회한다. 시간·자치구를 바꿔도 바로 다시 조회한다.
  // 페이지가 열리면 지도를 만들고 첫 조회를 한다.
  ui.filters.addEventListener("submit", (event) => { event.preventDefault(); refresh(); });
  ui.hours.addEventListener("change", refresh);
  ui.district.addEventListener("change", refresh);
  initMap();
  refresh();
})();
