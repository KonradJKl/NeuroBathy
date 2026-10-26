// The 3D view of the project page: the point cloud of the best model with the survey
// points. README.md describes the files in data/.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DATA_DIR = 'data/';
const BACKGROUND = '#11151b';
// matplotlib's viridis and coolwarm in 11 even steps, the colours of the figures
const VIRIDIS = ['#440154', '#482475', '#414487', '#355f8d', '#2a788e', '#21918c',
  '#22a884', '#44bf70', '#7ad151', '#bddf26', '#fde725'];
const COOLWARM = ['#3b4cc0', '#5977e3', '#7b9ff9', '#9ebeff', '#c0d4f5', '#dddcdc',
  '#f2cbb7', '#f7ac8e', '#ee8468', '#d65244', '#b40426'];
const NO_PREDICTION = '#8c959f';
const ERROR_LIMIT = 1;  // m, the ends of the error colours
const LAND_BRIGHTNESS = 0.55;  // land points while the riverbed shows depth
const POINT_SPREAD = 1.9;  // point diameter in mean point spacings
const MAX_POINT_SIZE = 12;  // CSS px
const MARKER_SIZE = { small: 8, large: 11 };  // CSS px, below and above 600 px width
const PICK_RADIUS = { mouse: 9, pen: 12, touch: 22 };  // CSS px
const HOME_ELEVATION = 50;  // degrees above the horizon
const HOME_TURN = 20;  // degrees away from looking straight across or along the river
const KINDS = { target: 'Underwater reference target', transect: 'Cross-section point' };

const CLOUD_VERTEX = /* glsl */ `
  attribute float depthCode;
  uniform float pointSize;
  uniform float maxPointSize;
  uniform float pixelsPerUnit;
  uniform float depthColouring;
  uniform float landBrightness;
  uniform float exaggeration;
  uniform float depthUnits;  // depth_max in position units
  uniform vec3 viridis[11];
  varying vec3 vColour;

  vec3 depthColour(float t) {
    float x = clamp(t, 0.0, 1.0) * 10.0;
    int i = int(min(floor(x), 9.0));
    return mix(viridis[i], viridis[i + 1], x - float(i));
  }

  void main() {
    float code = floor(depthCode * 255.0 + 0.5);
    float depth = code < 0.5 ? 0.0 : (code - 1.0) / 254.0;  // fraction of depth_max
    float z = position.z - (exaggeration - 1.0) * depth * depthUnits;
    vec4 viewPosition = modelViewMatrix * vec4(position.xy, z, 1.0);
    gl_Position = projectionMatrix * viewPosition;
    gl_PointSize = clamp(pointSize * pixelsPerUnit / -viewPosition.z, 1.0, maxPointSize);
    if (depthColouring < 0.5) vColour = color;
    else if (code < 0.5) vColour = color * landBrightness;
    else vColour = depthColour(depth);
  }
`;

const CLOUD_FRAGMENT = /* glsl */ `
  varying vec3 vColour;

  void main() {
    vec2 offset = gl_PointCoord - 0.5;
    if (dot(offset, offset) > 0.25) discard;
    gl_FragColor = vec4(vColour, 1.0);
  }
`;

const MARKER_VERTEX = /* glsl */ `
  attribute float markerIndex;
  uniform float markerSize;
  uniform float selected;
  varying vec3 vColour;
  varying float vSelected;

  void main() {
    vColour = color;
    vSelected = abs(markerIndex - selected) < 0.5 ? 1.0 : 0.0;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = markerSize * (1.0 + 0.5 * vSelected);
  }
`;

const MARKER_FRAGMENT = /* glsl */ `
  varying vec3 vColour;
  varying float vSelected;

  void main() {
    float r = 2.0 * length(gl_PointCoord - 0.5);
    float edge = fwidth(r);
    float alpha = 1.0 - smoothstep(1.0 - edge, 1.0, r);
    if (alpha <= 0.0) discard;
    vec3 ring = mix(vec3(0.07, 0.08, 0.1), vec3(1.0), vSelected);
    gl_FragColor = vec4(mix(vColour, ring, smoothstep(0.66 - edge, 0.66, r)), alpha);
  }
`;

const ui = {
  stage: document.getElementById('viewer-stage'),
  status: document.getElementById('viewer-status'),
  hint: document.getElementById('viewer-hint'),
  legend: document.getElementById('viewer-legend'),
  attribution: document.getElementById('viewer-attribution'),
  tooltip: document.getElementById('viewer-tooltip'),
  message: document.getElementById('viewer-message'),
  controls: document.getElementById('viewer-controls'),
  colouring: [...document.querySelectorAll('[data-colouring]')],
  survey: document.getElementById('viewer-survey'),
  exaggeration: document.getElementById('viewer-exaggeration'),
  exaggerationValue: document.getElementById('viewer-exaggeration-value'),
  reset: document.getElementById('viewer-reset'),
};

const settings = { colouring: 'rgb', survey: true, exaggeration: 1 };

start();

async function start() {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('webgl2', { antialias: false, powerPreference: 'high-performance' });
  if (!context) {
    showMessage('The 3D view needs WebGL 2, which this browser does not provide.');
    return;
  }
  const view = createView(canvas, context);
  bindControls(view);
  bindPointer(view);
  try {
    await loadScene(view);
  } catch (error) {
    if (view.clouds.length > 0) setStatus(`Preview only. ${error.message}`);
    else showMessage(error.message);
  }
}

function createView(canvas, context) {
  const renderer = new THREE.WebGLRenderer({ canvas, context });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(BACKGROUND);
  canvas.setAttribute('aria-label', 'Point cloud of the river');
  ui.stage.prepend(canvas);

  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 5000);
  camera.up.set(0, 0, 1);
  camera.position.set(0, -1, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.screenSpacePanning = false;
  controls.zoomToCursor = true;
  controls.maxPolarAngle = 0.495 * Math.PI;
  controls.addEventListener('start', () => { ui.hint.hidden = true; });

  const view = {
    renderer,
    camera,
    controls,
    scene: new THREE.Scene(),
    cloudMaterial: createCloudMaterial(),
    clouds: [],
    markers: null,
    survey: [],
    meta: null,
    frame: null,
    selected: -1,
    needsRender: true,
  };
  ui.exaggeration.value = settings.exaggeration;
  new ResizeObserver(() => resize(view)).observe(ui.stage);
  renderer.setAnimationLoop(() => renderFrame(view));
  return view;
}

function createCloudMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      pointSize: { value: 0.1 },
      maxPointSize: { value: MAX_POINT_SIZE },
      pixelsPerUnit: { value: 1 },
      depthColouring: { value: 0 },
      landBrightness: { value: LAND_BRIGHTNESS },
      exaggeration: { value: 1 },
      depthUnits: { value: 0 },
      viridis: { value: VIRIDIS.map((hex) => new THREE.Vector3(...hexToRgb(hex))) },
    },
    vertexShader: CLOUD_VERTEX,
    fragmentShader: CLOUD_FRAGMENT,
    vertexColors: true,
  });
}

// Renders only when the camera moved or something changed.
function renderFrame(view) {
  const moved = view.controls.update();
  if (!moved && !view.needsRender) return;
  view.needsRender = false;
  view.renderer.render(view.scene, view.camera);
  placeTooltip(view);
}

function resize(view) {
  const width = ui.stage.clientWidth;
  const height = ui.stage.clientHeight;
  if (width === 0 || height === 0) return;
  const { renderer, camera } = view;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  const ratio = renderer.getPixelRatio();
  const uniforms = view.cloudMaterial.uniforms;
  const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
  uniforms.pixelsPerUnit.value = height * ratio / (2 * Math.tan(halfFov));
  uniforms.maxPointSize.value = MAX_POINT_SIZE * ratio;
  if (view.markers) view.markers.material.uniforms.markerSize.value = markerSize(view);
  view.needsRender = true;
}

function markerSize(view) {
  const size = ui.stage.clientWidth < 600 ? MARKER_SIZE.small : MARKER_SIZE.large;
  return size * view.renderer.getPixelRatio();
}

async function loadScene(view) {
  setStatus('Loading the point cloud');
  const meta = await fetchJson(DATA_DIR + 'scene.json');
  view.meta = meta;
  view.cloudMaterial.uniforms.depthUnits.value = meta.depth_max / meta.step;
  showAttribution(meta);
  const survey = fetchJson(DATA_DIR + meta.survey).then(
    (points) => ({ points }),
    (error) => ({ error }),
  );

  const total = meta.levels.reduce((sum, level) => sum + level.count, 0);
  let loaded = 0;
  let warning = '';
  for (const [i, level] of meta.levels.entries()) {
    const label = i === 0 ? 'Loading the point cloud' : `Loading all ${formatCount(total)} points`;
    const progress = (fraction) => setStatus(`${label}, ${Math.floor(100 * fraction)}%`);
    const points = await fetchLevel(level, progress);
    if (i === 0) {
      view.frame = sceneFrame(points, meta);
      resetView(view);
      showHint();
    }
    addCloud(view, points);
    loaded += level.count;
    setPointSize(view, loaded);
    if (i === 0) {
      const result = await survey;
      if (result.error) warning = result.error.message;
      else addMarkers(view, result.points);
    }
  }
  if (warning) setStatus(warning);
  else hideStatus();
}

async function fetchJson(url) {
  const response = await request(url);
  try {
    return await response.json();
  } catch {
    throw new Error(`${url} is not valid JSON.`);
  }
}

async function fetchLevel(level, onProgress) {
  const n = level.count;
  const bytes = await fetchBytes(DATA_DIR + level.file, 10 * n, onProgress);
  // Typed arrays read in the byte order of the platform, which is little-endian
  // wherever browsers run, like the files.
  return {
    count: n,
    positions: new Uint16Array(bytes.buffer, 0, 3 * n),
    colours: new Uint8Array(bytes.buffer, 6 * n, 3 * n),
    depthCodes: new Uint8Array(bytes.buffer, 9 * n, n),
  };
}

async function fetchBytes(url, size, onProgress) {
  const response = await request(url);
  const bytes = new Uint8Array(size);
  const reader = response.body.getReader();
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.length > size) throw new Error(`${url} is larger than scene.json says.`);
    bytes.set(value, received);
    received += value.length;
    onProgress(received / size);
  }
  if (received !== size) throw new Error(`${url} is smaller than scene.json says.`);
  return bytes;
}

async function request(url) {
  let response;
  try {
    response = await fetch(url);
  } catch {
    throw new Error(`Could not load ${url}.`);
  }
  if (!response.ok) throw new Error(`Could not load ${url} (HTTP ${response.status}).`);
  return response;
}

// Centre (on the water surface), direction and size of the river, from the water points
// of the first level, and the area the cloud covers, for the point size.
function sceneFrame(points, meta) {
  const water = samplePoints(points, meta, (i) => points.depthCodes[i] > 0);
  const sample = water.length >= 100 ? water : samplePoints(points, meta, () => true);
  const cx = mean(sample.map((p) => p[0]));
  const cy = mean(sample.map((p) => p[1]));
  const cz = quantile(sample.map((p) => p[2]), 0.5);

  let xx = 0;
  let yy = 0;
  let xy = 0;
  for (const [x, y] of sample) {
    xx += (x - cx) ** 2;
    yy += (y - cy) ** 2;
    xy += (x - cx) * (y - cy);
  }
  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  const axis = new THREE.Vector2(Math.cos(angle), Math.sin(angle));
  const along = sample.map(([x, y]) => (x - cx) * axis.x + (y - cy) * axis.y);
  const across = sample.map(([x, y]) => (y - cy) * axis.x - (x - cx) * axis.y);
  return {
    centre: new THREE.Vector3(cx, cy, cz),
    axis,
    length: quantile(along, 0.98) - quantile(along, 0.02),
    width: quantile(across, 0.98) - quantile(across, 0.02),
    area: coveredArea(points, meta.step),
  };
}

// Every n-th point that passes keep(i), with the height of the water surface above it
function samplePoints(points, meta, keep) {
  const every = Math.max(1, Math.floor(points.count / 50000));
  const { step, depth_max: depthMax } = meta;
  const p = points.positions;
  const sample = [];
  for (let i = 0; i < points.count; i += every) {
    if (!keep(i)) continue;
    const code = points.depthCodes[i];
    const depth = code === 0 ? 0 : (code - 1) / 254 * depthMax;
    sample.push([p[3 * i] * step, p[3 * i + 1] * step, p[3 * i + 2] * step + depth]);
  }
  return sample;
}

// Square metres of 1 m cells that hold at least one point
function coveredArea(points, step) {
  const cells = new Set();
  const p = points.positions;
  for (let i = 0; i < points.count; i++) {
    cells.add(Math.floor(p[3 * i] * step) * 65536 + Math.floor(p[3 * i + 1] * step));
  }
  return cells.size;
}

// Looks across the river, or along it where that frames the river closer (portrait screens).
function homePose(frame, camera) {
  const elevation = THREE.MathUtils.degToRad(HOME_ELEVATION);
  const tanVertical = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const tanHorizontal = tanVertical * camera.aspect;
  const fit = (wide, deep) => 1.3 * Math.max(
    wide / 2 / tanHorizontal,
    deep * Math.sin(elevation) / 2 / tanVertical,
  );
  const across = fit(frame.length, frame.width);
  const along = fit(frame.width, frame.length);
  const direction = across <= along ? new THREE.Vector2(-frame.axis.y, frame.axis.x) : frame.axis.clone();
  if (direction.y < 0) direction.negate();  // look north, as on a map
  direction.rotateAround(new THREE.Vector2(), THREE.MathUtils.degToRad(HOME_TURN));
  const distance = Math.min(across, along);
  const target = frame.centre.clone();
  const position = new THREE.Vector3(
    target.x - direction.x * distance * Math.cos(elevation),
    target.y - direction.y * distance * Math.cos(elevation),
    target.z + distance * Math.sin(elevation),
  );
  return { position, target, distance };
}

function resetView(view) {
  if (!view.frame) return;
  const { camera, controls } = view;
  const { position, target, distance } = homePose(view.frame, camera);
  camera.position.copy(position);
  camera.near = distance / 500;
  camera.far = distance * 20;
  camera.updateProjectionMatrix();
  controls.target.copy(target);
  controls.minDistance = 1;
  controls.maxDistance = distance * 4;
  controls.update();
  view.needsRender = true;
}

function addCloud(view, points) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(points.positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(points.colours, 3, true));
  geometry.setAttribute('depthCode', new THREE.BufferAttribute(points.depthCodes, 1, true));
  const cloud = new THREE.Points(geometry, view.cloudMaterial);
  cloud.scale.setScalar(view.meta.step);
  cloud.frustumCulled = false;
  view.clouds.push(cloud);
  view.scene.add(cloud);
}

function setPointSize(view, count) {
  const spacing = Math.sqrt(view.frame.area / count);
  view.cloudMaterial.uniforms.pointSize.value = POINT_SPREAD * spacing;
  view.needsRender = true;
}

function addMarkers(view, survey) {
  const positions = new Float32Array(3 * survey.length);
  const colours = new Float32Array(3 * survey.length);
  const indices = new Float32Array(survey.length);
  survey.forEach((point, i) => {
    positions.set([point.x, point.y, markerHeight(point, settings.exaggeration)], 3 * i);
    colours.set(markerColour(point), 3 * i);
    indices[i] = i;
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colours, 3));
  geometry.setAttribute('markerIndex', new THREE.BufferAttribute(indices, 1));
  // drawn last and without depth test, so riverbed points never hide them
  const material = new THREE.ShaderMaterial({
    uniforms: {
      markerSize: { value: markerSize(view) },
      selected: { value: -1 },
    },
    vertexShader: MARKER_VERTEX,
    fragmentShader: MARKER_FRAGMENT,
    vertexColors: true,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  view.survey = survey;
  view.markers = new THREE.Points(geometry, material);
  view.markers.frustumCulled = false;
  view.markers.renderOrder = 1;
  view.markers.visible = settings.survey;
  view.scene.add(view.markers);
  updateLegend(view);
}

function markerColour(point) {
  if (point.depth_pred === null) return hexToRgb(NO_PREDICTION);
  const error = point.depth_pred - point.depth_gt;
  return rampColour(COOLWARM, (error + ERROR_LIMIT) / (2 * ERROR_LIMIT));
}

function bindControls(view) {
  for (const button of ui.colouring) {
    button.addEventListener('click', () => {
      settings.colouring = button.dataset.colouring;
      applyColouring(view);
    });
  }
  ui.survey.addEventListener('click', () => {
    settings.survey = !settings.survey;
    applySurvey(view);
  });
  ui.exaggeration.addEventListener('input', () => {
    settings.exaggeration = Number(ui.exaggeration.value);
    applyExaggeration(view);
  });
  ui.reset.addEventListener('click', () => resetView(view));
}

function applyColouring(view) {
  for (const button of ui.colouring) {
    button.setAttribute('aria-pressed', String(button.dataset.colouring === settings.colouring));
  }
  view.cloudMaterial.uniforms.depthColouring.value = settings.colouring === 'depth' ? 1 : 0;
  updateLegend(view);
  view.needsRender = true;
}

function applySurvey(view) {
  ui.survey.setAttribute('aria-pressed', String(settings.survey));
  if (view.markers) view.markers.visible = settings.survey;
  if (!settings.survey) select(view, -1);
  updateLegend(view);
  view.needsRender = true;
}

// Deepens the river: water and survey points move down by (factor - 1) times their depth
// below the water surface, so land, trees and the waterline stay where they are.
function applyExaggeration(view) {
  const factor = settings.exaggeration;
  ui.exaggerationValue.textContent = `${factor}×`;
  view.cloudMaterial.uniforms.exaggeration.value = factor;
  if (view.markers) {
    const positions = view.markers.geometry.attributes.position;
    view.survey.forEach((point, i) => positions.setZ(i, markerHeight(point, factor)));
    positions.needsUpdate = true;
  }
  view.needsRender = true;
}

function markerHeight(point, factor) {
  return point.z - (factor - 1) * Math.max(point.depth_gt, 0);
}

function bindPointer(view) {
  const canvas = view.renderer.domElement;
  let press = null;
  canvas.addEventListener('pointerdown', (event) => {
    press = { x: event.clientX, y: event.clientY, time: event.timeStamp };
  });
  canvas.addEventListener('pointerup', (event) => {
    if (!press) return;
    const moved = Math.hypot(event.clientX - press.x, event.clientY - press.y);
    const tap = moved < 6 && event.timeStamp - press.time < 500;
    press = null;
    if (tap) select(view, pickMarker(view, event));
  });
  canvas.addEventListener('pointermove', (event) => {
    if (event.pointerType !== 'mouse' || event.buttons !== 0) return;
    const index = pickMarker(view, event);
    canvas.style.cursor = index >= 0 ? 'pointer' : '';
    if (index !== view.selected) select(view, index);
  });
  canvas.addEventListener('pointerleave', (event) => {
    if (event.pointerType === 'mouse') select(view, -1);
  });
}

function pickMarker(view, event) {
  if (!view.markers || !view.markers.visible) return -1;
  const rect = view.renderer.domElement.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let best = -1;
  let bestDistance = PICK_RADIUS[event.pointerType] ?? PICK_RADIUS.mouse;
  for (let i = 0; i < view.survey.length; i++) {
    const screen = markerOnScreen(view, i, rect);
    if (!screen) continue;
    const distance = Math.hypot(screen.x - x, screen.y - y);
    if (distance < bestDistance) {
      best = i;
      bestDistance = distance;
    }
  }
  return best;
}

const projected = new THREE.Vector3();

function markerOnScreen(view, index, rect) {
  projected.fromBufferAttribute(view.markers.geometry.attributes.position, index).project(view.camera);
  if (Math.abs(projected.z) > 1) return null;
  return { x: (projected.x + 1) / 2 * rect.width, y: (1 - projected.y) / 2 * rect.height };
}

function select(view, index) {
  view.selected = index;
  if (view.markers) view.markers.material.uniforms.selected.value = index;
  if (index >= 0) fillTooltip(view.survey[index]);
  else ui.tooltip.hidden = true;
  view.needsRender = true;
}

function fillTooltip(point) {
  const rows = [['Surveyed depth', formatMetres(point.depth_gt)]];
  if (point.depth_pred === null) {
    rows.push(['Predicted depth', 'none']);
  } else {
    rows.push(['Predicted depth', formatMetres(point.depth_pred)]);
    rows.push(['Error', formatMetres(point.depth_pred - point.depth_gt, true)]);
  }
  const list = document.createElement('dl');
  for (const [label, value] of rows) list.append(element('dt', label), element('dd', value));
  const kind = element('span', KINDS[point.kind] ?? point.kind, 'kind');
  ui.tooltip.replaceChildren(element('strong', point.name), kind, list);
}

function placeTooltip(view) {
  if (view.selected < 0) return;
  const rect = view.renderer.domElement.getBoundingClientRect();
  const screen = markerOnScreen(view, view.selected, rect);
  ui.tooltip.hidden = !screen;
  if (!screen) return;
  const { offsetWidth: width, offsetHeight: height } = ui.tooltip;
  const left = Math.min(Math.max(screen.x - width / 2, 8), rect.width - width - 8);
  const top = screen.y - height - 14 >= 8 ? screen.y - height - 14 : screen.y + 14;
  ui.tooltip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
}

function updateLegend(view) {
  const blocks = [];
  if (settings.colouring === 'depth' && view.meta) {
    const depthMax = `${Number(view.meta.depth_max.toFixed(2))} m`;
    blocks.push(legendBlock('Depth below the water surface', VIRIDIS, ['0 m', depthMax]));
  }
  if (settings.survey && view.markers) {
    const ticks = [`−${ERROR_LIMIT} m`, '0', `+${ERROR_LIMIT} m`];
    const block = legendBlock('Survey points, predicted minus surveyed depth', COOLWARM, ticks);
    if (view.survey.some((point) => point.depth_pred === null)) {
      block.append(element('div', 'Grey: no prediction', 'legend-ticks'));
    }
    blocks.push(block);
  }
  ui.legend.replaceChildren(...blocks);
  ui.legend.hidden = blocks.length === 0;
}

function legendBlock(title, stops, ticks) {
  const bar = element('div', '', 'legend-bar');
  bar.style.background = `linear-gradient(to right, ${stops.join(', ')})`;
  const tickRow = element('div', '', 'legend-ticks');
  tickRow.append(...ticks.map((tick) => element('span', tick)));
  const block = document.createElement('div');
  block.append(element('div', title), bar, tickRow);
  return block;
}

function showAttribution(meta) {
  let text = meta.attribution || '';
  if (meta.license && !text.includes(meta.license)) text += `${text ? ', ' : ''}${meta.license}`;
  const parts = text.split(/(https?:\/\/\S+)/);
  ui.attribution.replaceChildren(...parts.map((part, i) => (i % 2 ? link(part) : part)));
  ui.attribution.hidden = text === '';
}

function showHint() {
  const touch = window.matchMedia('(pointer: coarse)').matches;
  ui.hint.textContent = touch
    ? 'Drag to rotate, pinch to zoom, two fingers to pan'
    : 'Drag to rotate, right-drag to pan, scroll to zoom';
  ui.hint.hidden = false;
}

function setStatus(text) {
  ui.status.textContent = text;
  ui.status.hidden = false;
}

function hideStatus() {
  ui.status.hidden = true;
}

function showMessage(text) {
  hideStatus();
  ui.controls.hidden = true;
  ui.message.textContent = text;
  ui.message.hidden = false;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function link(url) {
  const anchor = element('a', url.replace(/^https?:\/\//, ''));
  anchor.href = url;
  return anchor;
}

function formatCount(n) {
  return n.toLocaleString('en-US');
}

function formatMetres(value, signed = false) {
  const rounded = Math.round(value * 100) / 100;
  const sign = rounded < 0 ? '−' : signed && rounded > 0 ? '+' : '';
  return `${sign}${Math.abs(rounded).toFixed(2)} m`;
}

function hexToRgb(hex) {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
}

function rampColour(stops, t) {
  const x = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  const a = hexToRgb(stops[i]);
  const b = hexToRgb(stops[i + 1]);
  return a.map((value, k) => value + (b[k] - value) * (x - i));
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function quantile(values, q) {
  const sorted = Float64Array.from(values).sort();
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}
