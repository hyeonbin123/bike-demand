(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    "filters", "hours", "district", "refresh", "district-message", "snapshot-time",
    "health-message", "risk-note", "risk-status", "map-count", "map-message",
    "station-count", "risk-rows", "empty-message", "detail", "detail-title",
    "detail-snapshot", "detail-status", "forecast-bars", "detail-note",
    "detail-version", "risk-version",
  ].map((id) => [id, byId(id)]));
  const stampFormat = new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23",
  });
  const hourFormat = new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  });
  let map = null;
  let circles = null;
  let queryController = null;
  let detailController = null;
  const markers = new Map();
  const rowButtons = new Map();

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function stamp(value) {
    return value ? stampFormat.format(new Date(value)) : "없음";
  }

  function tier(value) {
    return value >= 10 ? "high" : value >= 5 ? "medium" : "low";
  }

  function shortfallText(value) {
    // API는 양수만 반환하지만 0.05 미만은 한 자리 반올림 뒤 0.0이 된다.
    return value === 0 ? "0.1 미만" : value.toFixed(1);
  }

  function hasCoordinates(station) {
    return Number.isFinite(station.lat) && Number.isFinite(station.lon)
      && Math.abs(station.lat) <= 90 && Math.abs(station.lon) <= 180;
  }

  async function request(url, signal) {
    const controller = new AbortController();
    const cancel = () => controller.abort();
    signal.addEventListener("abort", cancel, { once: true });
    if (signal.aborted) controller.abort();
    const timer = setTimeout(cancel, 15000);
    try {
      const response = await fetch(url, { signal: controller.signal, headers: { Accept: "application/json" } });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        const error = new Error("request failed");
        error.status = response.status;
        error.detail = body.detail;
        throw error;
      }
      return await response.json();
    } catch (error) {
      if (controller.signal.aborted && !signal.aborted) throw new Error("timeout");
      throw error;
    } finally {
      clearTimeout(timer);
      signal.removeEventListener("abort", cancel);
    }
  }

  function errorMessage(error) {
    if (error.status === 503) {
      if (error.detail === "no predictions") return "예측 데이터가 아직 없습니다. 예측이 준비된 뒤 새로고침해 주세요.";
      if (error.detail === "no recent snapshot") return "최근 30분 이내의 자전거 정보가 없습니다. 수집이 재개된 뒤 새로고침해 주세요.";
      return "서비스 데이터를 지금 조회할 수 없습니다. 잠시 후 새로고침해 주세요.";
    }
    if (error.status === 404) return "이 대여소를 찾을 수 없습니다. 목록을 새로고침해 주세요.";
    return "정보를 불러오지 못했습니다. 연결을 확인한 뒤 새로고침해 주세요.";
  }

  function initMap() {
    if (!window.L) {
      ui["map-message"].textContent = "지도 도구를 불러오지 못했습니다. 대여소 목록은 계속 사용할 수 있습니다.";
      return;
    }
    try {
      map = window.L.map("map", { scrollWheelZoom: false }).setView([37.5665, 126.9780], 11);
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

  function renderBars(predictions, requestedAt) {
    const values = new Map(predictions.map((item) => [new Date(item.hour_start).getTime(), item.predicted_rentals]));
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

  ui.filters.addEventListener("submit", (event) => { event.preventDefault(); refresh(); });
  ui.hours.addEventListener("change", refresh);
  ui.district.addEventListener("change", refresh);
  initMap();
  refresh();
})();
