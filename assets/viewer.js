// The 3D view of the project page: the point clouds of two models with the survey points,
// and on 0.1 m cells the riverbed of the model or of the LiDAR, or the model's depth minus
// the LiDAR depth. README.md describes the files in data/.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DATA_DIR = 'data/';
const BACKGROUND = '#11151b';
// matplotlib's viridis and coolwarm in 11 even steps, the colours of the figures
const VIRIDIS = ['#440154', '#482475', '#414487', '#355f8d', '#2a788e', '#21918c',
  '#22a884', '#44bf70', '#7ad151', '#bddf26', '#fde725'];
const COOLWARM = ['#3b4cc0', '#5977e3', '#7b9ff9', '#9ebeff', '#c0d4f5', '#dddcdc',
  '#f2cbb7', '#f7ac8e', '#ee8468', '#d65244', '#b40426'];
// survey points without a prediction, and cells without a LiDAR riverbed in Error
const GREY = '#8c959f';
// survey points over the LiDAR riverbed, grey so that the white ring of a selected one shows
const NEUTRAL = '#c9d1d9';
const ERROR_LIMIT = 1;  // m, the ends of the error colours
const NO_LIDAR = -32768;  // the error of the cells without a LiDAR riverbed in the error files
const DIMMED_BRIGHTNESS = 0.55;  // the points around the riverbed in Depth, LiDAR and Error
const POINT_SPREAD = 1.9;  // point diameter in mean point spacings
const CELL_SPREAD = Math.SQRT2;  // point diameter in cell sizes, so round points cover their cells
const MAX_POINT_SIZE = 12;  // CSS px
const MARKER_SIZE = { small: 8, large: 11 };  // CSS px, below and above 600 px width
const PICK_RADIUS = { mouse: 9, pen: 12, touch: 22 };  // CSS px
const HOME_ELEVATION = 50;  // degrees above the horizon
const HOME_TURN = 20;  // degrees away from looking straight across or along the river
const KINDS = { target: 'Underwater reference target', transect: 'Cross-section point' };
// The colourings that draw the riverbed as a layer of cells, with the name of the layer
// in the status line, the title of its legend and its bit in the flags of the points
// that it replaces. Error draws the cells of the depth grid.
const RIVERBEDS = {
  depth: { name: 'the depth grid', title: 'Depth below the water surface', bit: 1 },
  lidar: { name: 'the LiDAR riverbed', title: 'Depth of the LiDAR riverbed', bit: 2 },
  error: { name: 'the errors against the LiDAR', title: 'Predicted minus LiDAR depth', bit: 1 },
};

const CLOUD_VERTEX = /* glsl */ `
  attribute float depthCode;
  attribute float replaced;  // flags of the riverbed layers that replace the point
  uniform float pointSize;
  uniform float maxPointSize;
  uniform float pixelsPerUnit;
  uniform float dimmed;
  uniform float replacedBit;  // the bit of the layer on show, 0 when no layer is shown
  uniform float dimmedBrightness;
  uniform float exaggeration;
  uniform float depthUnits;  // depth_max in position units
  varying vec3 vColour;

  void main() {
    if (replacedBit > 0.5 && mod(floor(replaced / replacedBit), 2.0) > 0.5) {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);  // beyond the far plane, so not drawn
      return;
    }
    float code = floor(depthCode * 255.0 + 0.5);
    float depth = code < 0.5 ? 0.0 : (code - 1.0) / 254.0;  // fraction of depth_max
    float z = position.z - (exaggeration - 1.0) * depth * depthUnits;
    vec4 viewPosition = modelViewMatrix * vec4(position.xy, z, 1.0);
    gl_Position = projectionMatrix * viewPosition;
    gl_PointSize = clamp(pointSize * pixelsPerUnit / -viewPosition.z, 1.0, maxPointSize);
    vColour = dimmed < 0.5 ? color : color * dimmedBrightness;
  }
`;

// Colours a cell by its depth, or with ERRORS by its error against the LiDAR
const CELL_VERTEX = /* glsl */ `
  attribute float depth;  // mm below the water surface
  uniform float pointSize;
  uniform float maxPointSize;
  uniform float pixelsPerUnit;
  uniform float exaggeration;
  uniform float mmUnits;  // 1 mm in position units
  uniform float depthMax;  // mm, the end of the colours
  uniform vec3 colours[11];
  varying vec3 vColour;
  #ifdef ERRORS
  attribute float error;  // mm, predicted minus LiDAR depth
  uniform float noLidar;  // the error of the cells without a LiDAR riverbed
  uniform float errorLimit;  // mm, the ends of the colours
  uniform vec3 grey;
  #endif

  vec3 rampColour(float t) {
    float x = clamp(t, 0.0, 1.0) * 10.0;
    int i = int(min(floor(x), 9.0));
    return mix(colours[i], colours[i + 1], x - float(i));
  }

  void main() {
    float z = position.z - (exaggeration - 1.0) * depth * mmUnits;
    vec4 viewPosition = modelViewMatrix * vec4(position.xy, z, 1.0);
    gl_Position = projectionMatrix * viewPosition;
    gl_PointSize = clamp(pointSize * pixelsPerUnit / -viewPosition.z, 1.0, maxPointSize);
    #ifdef ERRORS
    vColour = error == noLidar ? grey : rampColour(0.5 + 0.5 * error / errorLimit);
    #else
    vColour = rampColour(depth / depthMax);
    #endif
  }
`;

const POINT_FRAGMENT = /* glsl */ `
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
  models: document.getElementById('viewer-models'),
  model: [...document.querySelectorAll('[data-model]')],
  controls: document.getElementById('viewer-controls'),
  colouring: [...document.querySelectorAll('[data-colouring]')],
  survey: document.getElementById('viewer-survey'),
  exaggeration: document.getElementById('viewer-exaggeration'),
  exaggerationValue: document.getElementById('viewer-exaggeration-value'),
  reset: document.getElementById('viewer-reset'),
};

// model is the id of the chosen model, null for the first one of scene.json
const settings = { model: null, colouring: 'rgb', survey: true, exaggeration: 1 };

start();

function start() {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('webgl2', { antialias: false, powerPreference: 'high-performance' });
  if (!context) {
    showMessage('The 3D view needs WebGL 2, which this browser does not provide.');
    return;
  }
  const view = createView(canvas, context);
  bindControls(view);
  bindPointer(view);
  // the other model and the riverbed layers load after the scene, see requestModel and
  // requestRiverbed
  view.loading = loadScene(view).catch((error) => {
    if (view.model) showWarning(view, `Preview only. ${error.message}`);
    else showMessage(error.message);
  });
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

  // the uniforms that the clouds and the riverbed layers share
  const uniforms = {
    maxPointSize: { value: MAX_POINT_SIZE },
    pixelsPerUnit: { value: 1 },
    exaggeration: { value: 1 },
  };
  const view = {
    renderer,
    camera,
    controls,
    scene: new THREE.Scene(),
    uniforms,
    meta: null,
    origin: null,  // the origin of the first model, the frame of the view
    models: new Map(),  // id of a model -> its state, from its first use
    model: null,  // the state of the model on show
    layers: new Map(),  // riverbed layer of scene.json -> its points, null while loading
    markerMaterial: null,
    loading: null,
    warning: '',  // a load problem that the status line keeps showing
    selected: -1,
    needsRender: true,
  };
  ui.exaggeration.value = settings.exaggeration;
  new ResizeObserver(() => resize(view)).observe(ui.stage);
  renderer.setAnimationLoop(() => renderFrame(view));
  return view;
}

function createCloudMaterial(uniforms) {
  return new THREE.ShaderMaterial({
    uniforms: {
      ...uniforms,
      pointSize: { value: 0.1 },
      dimmed: { value: 0 },
      replacedBit: { value: 0 },
      dimmedBrightness: { value: DIMMED_BRIGHTNESS },
      depthUnits: { value: 0 },
    },
    vertexShader: CLOUD_VERTEX,
    fragmentShader: POINT_FRAGMENT,
    vertexColors: true,
  });
}

function createCellMaterial(view, layer, stops = VIRIDIS) {
  return new THREE.ShaderMaterial({
    uniforms: {
      ...view.uniforms,
      pointSize: { value: CELL_SPREAD * layer.cell_size },
      mmUnits: { value: 0.001 / layer.step },
      depthMax: { value: 1000 * view.meta.depth_max },
      colours: { value: stops.map((hex) => new THREE.Vector3(...hexToRgb(hex))) },
    },
    vertexShader: CELL_VERTEX,
    fragmentShader: POINT_FRAGMENT,
  });
}

// The cells of a grid coloured by their errors against the LiDAR
function createErrorMaterial(view, grid) {
  const material = createCellMaterial(view, grid, COOLWARM);
  material.defines = { ERRORS: '' };
  Object.assign(material.uniforms, {
    noLidar: { value: NO_LIDAR },
    errorLimit: { value: 1000 * ERROR_LIMIT },
    grey: { value: new THREE.Vector3(...hexToRgb(GREY)) },
  });
  return material;
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
  const { renderer, camera, uniforms } = view;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  const ratio = renderer.getPixelRatio();
  const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
  uniforms.pixelsPerUnit.value = height * ratio / (2 * Math.tan(halfFov));
  uniforms.maxPointSize.value = MAX_POINT_SIZE * ratio;
  if (view.markerMaterial) view.markerMaterial.uniforms.markerSize.value = markerSize(view);
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
  view.origin = meta.models[0].origin;
  showAttribution(meta);
  // only the models of scene.json can be chosen, by default the first one
  const ids = meta.models.map((entry) => entry.id);
  for (const button of ui.model) button.hidden = !ids.includes(button.dataset.model);
  if (!ids.includes(settings.model)) settings.model = ids[0];
  await loadModel(view, addModel(view, meta.models[0]));
  applyModel(view);  // a model chosen while the first one loaded
}

// The state of a model of scene.json in the view. Its objects are moved by its origin
// minus the origin of the first model.
function addModel(view, entry) {
  const material = createCloudMaterial(view.uniforms);
  material.uniforms.depthUnits.value = view.meta.depth_max / entry.step;
  const model = {
    entry,
    offset: new THREE.Vector3(...entry.origin.map((value, k) => value - view.origin[k])),
    material,
    clouds: [],
    frame: null,
    survey: [],
    markers: null,
    replaced: 'none',  // the flags of the points: none, loading or marked
    shown: false,  // on show before, so a switch to it keeps the camera
  };
  view.models.set(entry.id, model);
  return model;
}

// Loads the levels and the survey points of a model. Its first level goes on show at
// once if the model is still the chosen one, or if nothing is on show yet.
async function loadModel(view, model) {
  const { entry } = model;
  const survey = fetchJson(DATA_DIR + entry.survey).then(
    (points) => ({ points }),
    (error) => ({ error }),
  );

  const total = entry.levels.reduce((sum, level) => sum + level.count, 0);
  let loaded = 0;
  let warning = '';
  for (const [i, level] of entry.levels.entries()) {
    const label = i === 0 ? 'Loading the point cloud' : `Loading all ${formatCount(total)} points`;
    const progress = (fraction) => setStatus(`${label}, ${Math.floor(100 * fraction)}%`);
    const points = await fetchLevel(level, progress);
    if (i === 0) model.frame = sceneFrame(points, model, view.meta.depth_max);
    addCloud(view, model, points);
    loaded += level.count;
    setPointSize(view, model, loaded);
    if (i === 0) {
      if (!view.model || settings.model === entry.id) showModel(view, model);
      const result = await survey;
      if (result.error) warning = result.error.message;
      else addMarkers(view, model, result.points);
    }
  }
  showWarning(view, warning || view.warning);
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
// of the first level, and the area the cloud covers, for the point size. When the home
// view of the model frames its whole cloud, the centre and size are those of all points,
// still measured along the river. The centre is in the frame of the view.
function sceneFrame(points, model, depthMax) {
  const { step, home } = model.entry;
  const water = samplePoints(points, step, depthMax, (i) => points.depthCodes[i] > 0);
  const river = water.length >= 100 ? water : samplePoints(points, step, depthMax, () => true);
  const sample = home === 'river' ? river : samplePoints(points, step, depthMax, () => true);
  const cx = mean(sample.map((p) => p[0]));
  const cy = mean(sample.map((p) => p[1]));
  const cz = quantile(sample.map((p) => p[2]), 0.5);
  const axis = principalAxis(river);
  const along = sample.map(([x, y]) => (x - cx) * axis.x + (y - cy) * axis.y);
  const across = sample.map(([x, y]) => (y - cy) * axis.x - (x - cx) * axis.y);
  return {
    centre: new THREE.Vector3(cx, cy, cz).add(model.offset),
    axis,
    length: quantile(along, 0.98) - quantile(along, 0.02),
    width: quantile(across, 0.98) - quantile(across, 0.02),
    area: coveredArea(points, step),
  };
}

// Direction of the longest extent of the points, seen from above
function principalAxis(sample) {
  const cx = mean(sample.map((p) => p[0]));
  const cy = mean(sample.map((p) => p[1]));
  let xx = 0;
  let yy = 0;
  let xy = 0;
  for (const [x, y] of sample) {
    xx += (x - cx) ** 2;
    yy += (y - cy) ** 2;
    xy += (x - cx) * (y - cy);
  }
  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  return new THREE.Vector2(Math.cos(angle), Math.sin(angle));
}

// Every n-th point that passes keep(i), with the height of the water surface above it
function samplePoints(points, step, depthMax, keep) {
  const every = Math.max(1, Math.floor(points.count / 50000));
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

// Moves the camera to the home view of the model on show.
function resetView(view) {
  if (!view.model) return;
  const { camera, controls } = view;
  const { position, target, distance } = homePose(view.model.frame, camera);
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

function addCloud(view, model, points) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(points.positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(points.colours, 3, true));
  geometry.setAttribute('depthCode', new THREE.BufferAttribute(points.depthCodes, 1, true));
  const cloud = new THREE.Points(geometry, model.material);
  cloud.position.copy(model.offset);
  cloud.scale.setScalar(model.entry.step);
  cloud.frustumCulled = false;
  cloud.visible = model === view.model;
  model.clouds.push(cloud);
  view.scene.add(cloud);
}

function setPointSize(view, model, count) {
  const spacing = Math.sqrt(model.frame.area / count);
  model.material.uniforms.pointSize.value = POINT_SPREAD * spacing;
  view.needsRender = true;
}

// Shows the chosen model, after loading it on its first use. Until its first level has
// loaded, the model before stays on show.
function applyModel(view) {
  for (const button of ui.model) {
    button.setAttribute('aria-pressed', String(button.dataset.model === settings.model));
  }
  if (!view.meta) return;  // applied once scene.json has arrived
  const model = view.models.get(settings.model) ?? requestModel(view, settings.model);
  if (model.clouds.length > 0) showModel(view, model);
}

// Loads a model on its first use, after the loads before it. If a load fails after its
// preview, the model stays a preview like the first one; if nothing of it came, the
// status line shows the error, the switch goes back to the model on show and the next
// click tries again.
function requestModel(view, id) {
  const model = addModel(view, view.meta.models.find((entry) => entry.id === id));
  view.loading = view.loading.then(async () => {
    try {
      await loadModel(view, model);
    } catch (error) {
      if (model.clouds.length > 0) {
        showWarning(view, `Preview only. ${error.message}`);
        return;
      }
      view.models.delete(id);
      if (view.model && settings.model === id) {
        settings.model = view.model.entry.id;
        applyModel(view);
      }
      setStatus(error.message);
    }
  });
  return model;
}

// Puts a model on show in place of the one before. The first time, the camera moves to
// its home view; later switches keep the camera.
function showModel(view, model) {
  if (model === view.model) return;
  if (!view.model) showHint();
  view.model = model;
  for (const other of view.models.values()) {
    for (const cloud of other.clouds) cloud.visible = other === model;
    if (other.markers) other.markers.visible = other === model && settings.survey;
  }
  select(view, -1);
  if (!model.shown) {
    model.shown = true;
    resetView(view);
  }
  applyColouring(view);
}

// The layer of scene.json that the current colouring draws as the riverbed, if any
function riverbedLayer(view) {
  if (!view.model || settings.colouring === 'rgb') return null;
  if (settings.colouring === 'lidar') return view.meta.lidar;
  return settings.colouring === 'error' ? view.model.entry.error : view.model.entry.grid;
}

// Loads what the current colouring draws on its first use: the flags of the points of
// the model on show and the layer, for Error the cells of the grid and their errors.
function requestRiverbed(view) {
  const { model } = view;
  const { name } = RIVERBEDS[settings.colouring];
  requestReplaced(view, model, name);
  const cells = settings.colouring === 'lidar' ? view.meta.lidar : model.entry.grid;
  requestLayer(view, cells, () => loadCells(view, cells, name));
  if (settings.colouring === 'error') {
    requestLayer(view, model.entry.error, () => loadErrors(view, model, name));
  }
}

// Loads a riverbed layer on its first use. Loads run one after the other, after the
// scene, so the status line shows one at a time.
function requestLayer(view, layer, load) {
  if (view.layers.has(layer)) return;
  view.layers.set(layer, null);
  view.loading = view.loading.then(async () => {
    try {
      view.layers.set(layer, await load());
      showWarning(view, view.warning);
      showRiverbed(view);
    } catch (error) {
      view.layers.delete(layer);  // so the next click tries again
      setStatus(error.message);
    }
  });
}

// Loads the flags of a model on their first use, in the same queue as the layers.
function requestReplaced(view, model, name) {
  if (model.replaced !== 'none') return;
  model.replaced = 'loading';
  view.loading = view.loading.then(async () => {
    try {
      setStatus(`Loading ${name}`);
      await markReplaced(view, model);
      showWarning(view, view.warning);
      showRiverbed(view);
    } catch (error) {
      model.replaced = 'none';  // so the next click tries again
      setStatus(error.message);
    }
  });
}

async function loadCells(view, layer, name) {
  setStatus(`Loading ${name}`);
  const n = layer.count;
  const progress = (fraction) => setStatus(`Loading ${name}, ${Math.floor(100 * fraction)}%`);
  const bytes = await fetchBytes(DATA_DIR + layer.file, 8 * n, progress);
  const positions = new Uint16Array(bytes.buffer, 0, 3 * n);
  const depths = new Uint16Array(bytes.buffer, 6 * n, n);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('depth', new THREE.BufferAttribute(depths, 1));
  const points = new THREE.Points(geometry, createCellMaterial(view, layer));
  points.position.fromArray(layer.origin.map((value, k) => value - view.origin[k]));
  points.scale.setScalar(layer.step);
  points.frustumCulled = false;
  view.scene.add(points);
  return points;
}

// The errors of the grid of a model, one int16 per cell, drawn on the cells of the grid
async function loadErrors(view, model, name) {
  const { grid, error } = model.entry;
  const cells = view.layers.get(grid);
  if (!cells) throw new Error(`Could not load ${DATA_DIR}${grid.file}.`);
  setStatus(`Loading ${name}`);
  const progress = (fraction) => setStatus(`Loading ${name}, ${Math.floor(100 * fraction)}%`);
  const bytes = await fetchBytes(DATA_DIR + error.file, 2 * grid.count, progress);
  cells.geometry.setAttribute('error', new THREE.BufferAttribute(new Int16Array(bytes.buffer), 1));
  const points = new THREE.Points(cells.geometry, createErrorMaterial(view, grid));
  points.position.copy(cells.position);
  points.scale.copy(cells.scale);
  points.frustumCulled = false;
  view.scene.add(points);
  return points;
}

// Flags the water points of the levels of a model that the riverbed layers replace, from
// one byte per point of the levels in order (bits as in RIVERBEDS).
async function markReplaced(view, model) {
  const total = model.entry.levels.reduce((sum, level) => sum + level.count, 0);
  const flags = await fetchBytes(DATA_DIR + model.entry.replaced, total, () => {});
  let offset = 0;
  for (const cloud of model.clouds) {
    const n = cloud.geometry.attributes.position.count;
    const part = flags.subarray(offset, offset + n);
    cloud.geometry.setAttribute('replaced', new THREE.BufferAttribute(part, 1));
    offset += n;
  }
  model.replaced = 'marked';
}

// Draws the layer of the current colouring in place of the water points that it
// replaces. Until it and the flags of the model have loaded, the water points stay,
// dimmed like the land.
function showRiverbed(view) {
  const layer = riverbedLayer(view);
  const shown = layer && view.model.replaced === 'marked' ? view.layers.get(layer) : null;
  for (const points of view.layers.values()) {
    if (points) points.visible = points === shown;
  }
  for (const model of view.models.values()) {
    const { uniforms } = model.material;
    uniforms.dimmed.value = settings.colouring === 'rgb' ? 0 : 1;
    uniforms.replacedBit.value = shown ? RIVERBEDS[settings.colouring].bit : 0;
  }
  view.needsRender = true;
}

function addMarkers(view, model, survey) {
  const positions = new Float32Array(3 * survey.length);
  const colours = new Float32Array(3 * survey.length);
  const indices = new Float32Array(survey.length);
  survey.forEach((point, i) => {
    positions.set([point.x, point.y, markerHeight(point, settings.exaggeration)], 3 * i);
    colours.set(markerColour(point, settings.colouring), 3 * i);
    indices[i] = i;
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colours, 3));
  geometry.setAttribute('markerIndex', new THREE.BufferAttribute(indices, 1));
  // drawn last and without depth test, so riverbed points never hide them
  view.markerMaterial ??= new THREE.ShaderMaterial({
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
  model.survey = survey;
  model.markers = new THREE.Points(geometry, view.markerMaterial);
  model.markers.position.copy(model.offset);
  model.markers.frustumCulled = false;
  model.markers.renderOrder = 1;
  model.markers.visible = model === view.model && settings.survey;
  view.scene.add(model.markers);
  updateLegend(view);
}

// The errors are the model's, so over the LiDAR riverbed every point is neutral.
function markerColour(point, colouring) {
  if (colouring === 'lidar') return hexToRgb(NEUTRAL);
  if (point.depth_pred === null) return hexToRgb(GREY);
  const error = point.depth_pred - point.depth_gt;
  return rampColour(COOLWARM, (error + ERROR_LIMIT) / (2 * ERROR_LIMIT));
}

function bindControls(view) {
  for (const button of ui.model) {
    button.addEventListener('click', () => {
      settings.model = button.dataset.model;
      applyModel(view);
    });
  }
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
  if (riverbedLayer(view)) requestRiverbed(view);
  showRiverbed(view);
  for (const model of view.models.values()) {
    if (!model.markers) continue;
    const colours = model.markers.geometry.attributes.color;
    model.survey.forEach((point, i) => colours.setXYZ(i, ...markerColour(point, settings.colouring)));
    colours.needsUpdate = true;
  }
  updateLegend(view);
}

function applySurvey(view) {
  ui.survey.setAttribute('aria-pressed', String(settings.survey));
  if (view.model?.markers) view.model.markers.visible = settings.survey;
  if (!settings.survey) select(view, -1);
  updateLegend(view);
  view.needsRender = true;
}

// Deepens the river: water points, riverbed cells and survey points move down by
// (factor - 1) times their depth below the water surface, so land, trees and the
// waterline stay where they are.
function applyExaggeration(view) {
  const factor = settings.exaggeration;
  ui.exaggerationValue.textContent = `${factor}×`;
  view.uniforms.exaggeration.value = factor;
  for (const model of view.models.values()) {
    if (!model.markers) continue;
    const positions = model.markers.geometry.attributes.position;
    model.survey.forEach((point, i) => positions.setZ(i, markerHeight(point, factor)));
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
  const markers = view.model?.markers;
  if (!markers || !markers.visible) return -1;
  const rect = view.renderer.domElement.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let best = -1;
  let bestDistance = PICK_RADIUS[event.pointerType] ?? PICK_RADIUS.mouse;
  for (let i = 0; i < view.model.survey.length; i++) {
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
  const { markers } = view.model;
  projected.fromBufferAttribute(markers.geometry.attributes.position, index);
  projected.add(markers.position).project(view.camera);
  if (Math.abs(projected.z) > 1) return null;
  return { x: (projected.x + 1) / 2 * rect.width, y: (1 - projected.y) / 2 * rect.height };
}

function select(view, index) {
  view.selected = index;
  if (view.markerMaterial) view.markerMaterial.uniforms.selected.value = index;
  if (index >= 0) fillTooltip(view.model.survey[index]);
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
  const riverbed = riverbedLegend(view);
  if (riverbed) blocks.push(riverbed);
  if (settings.survey && view.model?.markers && settings.colouring !== 'lidar') {
    const ticks = [`−${ERROR_LIMIT} m`, '0', `+${ERROR_LIMIT} m`];
    const block = legendBlock('Survey points, predicted minus surveyed depth', COOLWARM, ticks);
    if (view.model.survey.some((point) => point.depth_pred === null)) {
      block.append(element('div', 'Grey: no prediction', 'legend-ticks'));
    }
    blocks.push(block);
  }
  ui.legend.replaceChildren(...blocks);
  ui.legend.hidden = blocks.length === 0;
}

// The legend of the layer of the current colouring, with the RMSE of the model on show
// for Error
function riverbedLegend(view) {
  const riverbed = RIVERBEDS[settings.colouring];
  if (!riverbed || !view.model) return null;
  if (settings.colouring !== 'error') {
    const depthMax = `${Number(view.meta.depth_max.toFixed(2))} m`;
    return legendBlock(riverbed.title, VIRIDIS, ['0 m', depthMax]);
  }
  const ticks = [`≤ −${ERROR_LIMIT} m`, '0', `≥ +${ERROR_LIMIT} m`];
  const block = legendBlock(riverbed.title, COOLWARM, ticks);
  block.append(
    element('div', 'Grey: no LiDAR riverbed (dry ground or outside the LiDAR area)', 'legend-ticks'),
    element('div', `RMSE ${view.model.entry.error.rmse.toFixed(2)} m`, 'legend-ticks'),
  );
  return block;
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

// Shows a load problem in the status line, or hides the line.
function showWarning(view, warning) {
  view.warning = warning;
  if (warning) setStatus(warning);
  else hideStatus();
}

function showMessage(text) {
  hideStatus();
  ui.models.hidden = true;
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
