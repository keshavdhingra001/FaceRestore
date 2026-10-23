const CANVAS_SIZE = 512;

const AppState = {
    imageToken: 0,
    hasImage: false,
    engine: null,
    faceData: null,
    isProcessing: false,
    brushWidth: 18,
    brushMode: 'mask',
    mousePressed: false,
    lastX: 0,
    lastY: 0,
    isDrawingAllowed: false,
    compareValue: 50,
    isComparing: false
};

const History = {
    stack: [],
    step: -1,

    push(data) {
        this.step += 1;
        this.stack = this.stack.slice(0, this.step);
        this.stack.push(data);
        return this.step;
    },

    update(stepId, key, value) {
        const index = Number(stepId);
        if (this.stack[index]) {
            this.stack[index][key] = value;
        }
    },

    undo() {
        if (this.step > 0) {
            this.step -= 1;
        }
        return this.stack[this.step];
    },

    redo() {
        if (this.step < this.stack.length - 1) {
            this.step += 1;
        }
        return this.stack[this.step];
    },

    clear() {
        this.stack = [];
        this.step = -1;
    }
};

function byId(id) {
    return document.getElementById(id);
}

function showToast(message, type = 'success') {
    const container = byId('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(8px)';
        setTimeout(() => toast.remove(), 180);
    }, 3200);
}

function setStatus(message) {
    byId('status-line').textContent = message;
}

function setModelStatus(message) {
    byId('model-status').textContent = message;
}

function fillCanvas(canvas, img) {
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    if (img) {
        ctx.drawImage(img, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
    }
}

function drawImageCover(ctx, img, size) {
    const scale = Math.max(size / img.width, size / img.height);
    const width = img.width * scale;
    const height = img.height * scale;
    const x = (size - width) / 2;
    const y = (size - height) / 2;
    ctx.drawImage(img, x, y, width, height);
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('Image could not be loaded'));
        img.src = src;
    });
}

function dataURItoBlob(dataURI) {
    const binary = atob(dataURI.split(',')[1]);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
    }
    return new Blob([bytes], { type: 'image/png' });
}

function blobToDataURI(blob) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = event => resolve(event.target.result);
        reader.readAsDataURL(blob);
    });
}

async function requestJson(response) {
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || response.statusText);
    }
    if (data.error) {
        throw new Error(data.error);
    }
    return data;
}

const Api = {
    async engines() {
        return requestJson(await fetch('/api/engines'));
    },

    async inpaint(imageCanvas, maskCanvas, engine) {
        const formData = new FormData();
        formData.append('image', dataURItoBlob(imageCanvas.toDataURL('image/png')), 'image.png');
        formData.append('mask', dataURItoBlob(maskCanvas.toDataURL('image/png')), 'mask.png');
        formData.append('engine', engine || '');
        const response = await fetch('/api/inpaint', { method: 'POST', body: formData });
        if (!response.ok) {
            await requestJson(response);
        }
        return {
            result: await blobToDataURI(await response.blob()),
            faceFound: response.headers.get('X-Face-Found') === 'true'
        };
    },

    async detectFace(canvas) {
        const formData = new FormData();
        formData.append('image', dataURItoBlob(canvas.toDataURL('image/png')), 'image.png');
        return requestJson(await fetch('/api/detect', { method: 'POST', body: formData }));
    }
};

function computeMaskCoverage() {
    const canvas = byId('canvas-mask');
    const ctx = canvas.getContext('2d');
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let painted = 0;
    for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i + 3] > 0 && Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) > 0) {
            painted += 1;
        }
    }
    return painted / (canvas.width * canvas.height);
}

function updateMaskCoverage(coverage = computeMaskCoverage()) {
    const pct = Math.round(coverage * 100);
    byId('mask-coverage').textContent = `Mask ${pct}%`;
    if (coverage >= 0.995) {
        setStatus('Full mask: the model has no source pixels to copy');
    }
}

function updateCompare(value) {
    const nextValue = Math.max(0, Math.min(100, Number(value)));
    const compare = byId('canvas-compare-source');
    const handle = byId('compare-handle');
    const slider = byId('compare-slider');
    AppState.compareValue = nextValue;
    compare.style.clipPath = `inset(0 ${100 - nextValue}% 0 0)`;
    handle.style.left = `${nextValue}%`;
    if (slider && Number(slider.value) !== nextValue) {
        slider.value = String(nextValue);
    }
}

function setCompareFromPointer(event) {
    const shell = byId('result-shell');
    const rect = shell.getBoundingClientRect();
    const point = event.touches && event.touches.length ? event.touches[0] : event;
    const x = Math.max(0, Math.min(rect.width, point.clientX - rect.left));
    updateCompare(Math.round((x / rect.width) * 100));
}

function bindCompareDrag() {
    const shell = byId('result-shell');
    const slider = byId('compare-slider');

    slider.addEventListener('input', event => updateCompare(Number(event.target.value)));

    const beginCompare = (event) => {
        if (event.target.closest('.loading-overlay')) {
            return;
        }
        AppState.isComparing = true;
        if (event.pointerId !== undefined && shell.setPointerCapture) {
            shell.setPointerCapture(event.pointerId);
        }
        setCompareFromPointer(event);
    };

    const moveCompare = (event) => {
        if (!AppState.isComparing) {
            return;
        }
        event.preventDefault();
        setCompareFromPointer(event);
    };

    const endCompare = (event) => {
        AppState.isComparing = false;
        if (event.pointerId !== undefined && shell.hasPointerCapture && shell.hasPointerCapture(event.pointerId)) {
            shell.releasePointerCapture(event.pointerId);
        }
    };

    if (window.PointerEvent) {
        shell.addEventListener('pointerdown', beginCompare);
        shell.addEventListener('pointermove', moveCompare);
        shell.addEventListener('pointerup', endCompare);
        shell.addEventListener('pointercancel', endCompare);
    } else {
        shell.addEventListener('mousedown', beginCompare);
        document.addEventListener('mousemove', moveCompare);
        document.addEventListener('mouseup', endCompare);
        shell.addEventListener('touchstart', beginCompare, { passive: true });
        shell.addEventListener('touchmove', moveCompare, { passive: false });
        shell.addEventListener('touchend', endCompare);
    }
}

function setProcessing(active) {
    AppState.isProcessing = active;
    byId('loading-overlay').classList.toggle('active', active);
    byId('result-panel').classList.toggle('processing', active);
    setModelStatus(active ? 'Working' : 'Idle');
}

function setBrushMode(mode) {
    AppState.brushMode = mode;
    document.querySelectorAll('.tool-btn[data-mode]').forEach((button) => {
        button.classList.toggle('active', button.dataset.mode === mode);
    });
}

const DrawEngine = {
    init(maskCanvas, outCanvas) {
        this.maskCanvas = maskCanvas;
        this.maskCtx = maskCanvas.getContext('2d');
        this.outCanvas = outCanvas;
        this.bindEvents();
    },

    bindEvents() {
        const canvas = this.maskCanvas;
        canvas.addEventListener('mousedown', event => this.onStart(event));
        canvas.addEventListener('mousemove', event => this.onMove(event));
        canvas.addEventListener('mouseup', () => this.onEnd());
        canvas.addEventListener('mouseleave', () => this.onEnd());
        canvas.addEventListener('touchstart', event => {
            event.preventDefault();
            this.onStart(event);
        }, { passive: false });
        canvas.addEventListener('touchmove', event => {
            event.preventDefault();
            this.onMove(event);
        }, { passive: false });
        canvas.addEventListener('touchend', event => {
            event.preventDefault();
            this.onEnd();
        }, { passive: false });
    },

    getCoords(event) {
        const rect = this.maskCanvas.getBoundingClientRect();
        const point = event.touches && event.touches.length ? event.touches[0] : event;
        return {
            x: (point.clientX - rect.left) / rect.width * this.maskCanvas.width,
            y: (point.clientY - rect.top) / rect.height * this.maskCanvas.height
        };
    },

    withBrush(mode, draw) {
        this.maskCtx.save();
        this.maskCtx.globalCompositeOperation = mode === 'erase' ? 'destination-out' : 'source-over';
        this.maskCtx.fillStyle = 'white';
        this.maskCtx.strokeStyle = 'white';
        draw();
        this.maskCtx.restore();
    },

    onStart(event) {
        if (!AppState.isDrawingAllowed || AppState.isProcessing) {
            return;
        }
        AppState.mousePressed = true;
        const coords = this.getCoords(event);
        this.drawDot(coords.x, coords.y, AppState.brushMode);
        AppState.lastX = coords.x;
        AppState.lastY = coords.y;
        updateMaskCoverage();
    },

    onMove(event) {
        if (!AppState.mousePressed) {
            return;
        }
        const coords = this.getCoords(event);
        this.drawLine(AppState.lastX, AppState.lastY, coords.x, coords.y, AppState.brushMode);
        AppState.lastX = coords.x;
        AppState.lastY = coords.y;
        updateMaskCoverage();
    },

    onEnd() {
        if (!AppState.mousePressed) {
            return;
        }
        AppState.mousePressed = false;
        this.applyMask();
    },

    drawDot(x, y, mode = 'mask') {
        this.withBrush(mode, () => {
            this.maskCtx.beginPath();
            this.maskCtx.arc(x, y, AppState.brushWidth / 2, 0, Math.PI * 2);
            this.maskCtx.fill();
        });
    },

    drawLine(x1, y1, x2, y2, mode = 'mask') {
        this.withBrush(mode, () => {
            this.maskCtx.beginPath();
            this.maskCtx.lineWidth = AppState.brushWidth;
            this.maskCtx.lineJoin = 'round';
            this.maskCtx.lineCap = 'round';
            this.maskCtx.moveTo(x1, y1);
            this.maskCtx.lineTo(x2, y2);
            this.maskCtx.stroke();
        });
    },

    drawEllipse(cx, cy, rx, ry, rotation = 0, mode = 'mask') {
        this.withBrush(mode, () => {
            this.maskCtx.beginPath();
            this.maskCtx.ellipse(cx, cy, rx, ry, rotation, 0, Math.PI * 2);
            this.maskCtx.fill();
        });
        updateMaskCoverage();
    },

    async applyMask() {
        if (!AppState.hasImage || AppState.isProcessing) {
            return;
        }

        const token = AppState.imageToken;
        const coverage = computeMaskCoverage();
        const stepId = History.push({ mask: this.maskCanvas.toDataURL('image/png'), coverage });
        updateMaskCoverage(coverage);
        setProcessing(true);
        setStatus(coverage >= 0.995 ? 'Nothing left to copy from, filling from the face prior' : 'Restoring selected area');

        try {
            const data = await Api.inpaint(byId('canvas-in'), this.maskCanvas, AppState.engine);
            if (token !== AppState.imageToken) {
                return;
            }
            History.update(stepId, 'result', data.result);
            if (History.step === stepId) {
                const img = await loadImage(data.result);
                fillCanvas(this.outCanvas, img);
                setModelStatus('Ready');
            }
            setStatus(data.faceFound || coverage === 0 ? 'Result ready' : 'Result ready (no face found, used the whole image)');
        } catch (error) {
            showToast(`Restore failed: ${error.message}`, 'error');
            setStatus('Restore failed');
        } finally {
            setProcessing(false);
        }
    }
};

async function applyInitialState(img) {
    const canvasIn = byId('canvas-in');
    const canvasMask = byId('canvas-mask');
    const canvasOut = byId('canvas-out');
    const canvasCompare = byId('canvas-compare-source');

    fillCanvas(canvasIn, img);
    fillCanvas(canvasOut, img);
    fillCanvas(canvasCompare, img);
    fillCanvas(canvasMask, null);

    byId('empty-state').classList.add('hidden');
    byId('face-tools').classList.add('hidden');
    AppState.faceData = null;
    AppState.imageToken += 1;
    AppState.hasImage = true;
    AppState.isDrawingAllowed = true;

    History.clear();
    History.push({
        mask: canvasMask.toDataURL('image/png'),
        result: canvasIn.toDataURL('image/png'),
        coverage: 0
    });
    updateMaskCoverage(0);
    setStatus('Ready');
    detectFaceRegions(AppState.imageToken);
}

async function pickRandom() {
    setStatus('Loading sample image');
    try {
        const img = await loadImage(`/api/sample?t=${Date.now()}`);
        await applyInitialState(img);
    } catch (error) {
        showToast(`Sample failed: ${error.message}`, 'error');
        setStatus('Upload an image to begin');
    }
}

async function loadFromFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        showToast('Please choose an image file', 'error');
        return;
    }

    const dataURI = await blobToDataURI(file);
    const img = await loadImage(dataURI);
    const canvas = document.createElement('canvas');
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    drawImageCover(canvas.getContext('2d'), img, CANVAS_SIZE);
    const resized = await loadImage(canvas.toDataURL('image/png'));
    await applyInitialState(resized);
}

async function detectFaceRegions(token) {
    try {
        const data = await Api.detectFace(byId('canvas-in'));
        if (token !== AppState.imageToken) {
            return;
        }
        if (data.faces && data.faces.length > 0) {
            AppState.faceData = data.faces[0];
            byId('face-tools').classList.remove('hidden');
        }
    } catch (error) {
        console.warn('Face detection failed:', error);
    }
}

function applyFaceRegionMask(regionName) {
    if (!AppState.faceData || !AppState.isDrawingAllowed || AppState.isProcessing) {
        return;
    }
    const names = regionName.split(',');
    for (const name of names) {
        const r = AppState.faceData.regions[name];
        if (r) {
            DrawEngine.drawEllipse(r.cx, r.cy, r.rx, r.ry, r.angle * Math.PI / 180, 'mask');
        }
    }
    DrawEngine.applyMask();
}

async function restoreHistoryState(data) {
    if (!data) {
        return;
    }
    const maskImg = await loadImage(data.mask);
    fillCanvas(byId('canvas-mask'), maskImg);
    if (data.result) {
        const resultImg = await loadImage(data.result);
        fillCanvas(byId('canvas-out'), resultImg);
    }
    updateMaskCoverage(data.coverage);
}

async function doUndo() {
    await restoreHistoryState(History.undo());
}

async function doRedo() {
    await restoreHistoryState(History.redo());
}

async function doClear() {
    const canvasMask = byId('canvas-mask');
    canvasMask.getContext('2d').clearRect(0, 0, canvasMask.width, canvasMask.height);
    const source = await loadImage(byId('canvas-in').toDataURL('image/png'));
    fillCanvas(byId('canvas-out'), source);
    History.clear();
    History.push({
        mask: canvasMask.toDataURL('image/png'),
        result: byId('canvas-in').toDataURL('image/png'),
        coverage: 0
    });
    updateMaskCoverage(0);
    setStatus('Mask cleared');
}

function doFillMask() {
    if (!AppState.isDrawingAllowed || AppState.isProcessing) {
        return;
    }
    const canvasMask = byId('canvas-mask');
    const ctx = canvasMask.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvasMask.width, canvasMask.height);
    updateMaskCoverage(1);
    showToast('Full fill uses the face prior only', 'warning');
    DrawEngine.applyMask();
}

function saveResult() {
    if (!AppState.hasImage) {
        showToast('Load an image first', 'error');
        return;
    }
    const link = document.createElement('a');
    link.href = byId('canvas-out').toDataURL('image/png');
    link.download = 'facerestore_result.png';
    link.click();
}

function initDragDrop() {
    let dragCounter = 0;
    const dropZone = byId('drop-zone');

    document.addEventListener('dragenter', (event) => {
        event.preventDefault();
        dragCounter += 1;
        dropZone.classList.add('active');
    });

    document.addEventListener('dragleave', (event) => {
        event.preventDefault();
        dragCounter -= 1;
        if (dragCounter <= 0) {
            dragCounter = 0;
            dropZone.classList.remove('active');
        }
    });

    document.addEventListener('dragover', event => event.preventDefault());

    document.addEventListener('drop', (event) => {
        event.preventDefault();
        dragCounter = 0;
        dropZone.classList.remove('active');
        const file = event.dataTransfer.files[0];
        loadFromFile(file);
    });
}

async function loadEngines() {
    const select = byId('engine-select');
    try {
        const data = await Api.engines();
        select.innerHTML = '';
        for (const engine of data.engines) {
            const option = document.createElement('option');
            option.value = engine.id;
            option.textContent = engine.label;
            select.appendChild(option);
        }
        select.value = data.default;
        AppState.engine = data.default;
    } catch (error) {
        showToast(`Could not load models: ${error.message}`, 'error');
    }
    select.addEventListener('change', () => {
        AppState.engine = select.value;
        if (computeMaskCoverage() > 0) {
            DrawEngine.applyMask();
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    DrawEngine.init(byId('canvas-mask'), byId('canvas-out'));
    loadEngines();

    document.querySelectorAll('.btn-load').forEach((button) => {
        button.addEventListener('click', () => byId('file-upload').click());
    });
    byId('file-upload').addEventListener('change', (event) => {
        loadFromFile(event.target.files[0]);
        event.target.value = '';
    });

    document.querySelectorAll('.btn-pick-random').forEach((button) => {
        button.addEventListener('click', pickRandom);
    });
    document.querySelectorAll('.btn-save').forEach((button) => {
        button.addEventListener('click', saveResult);
    });

    document.querySelectorAll('.tool-btn[data-mode]').forEach((button) => {
        button.addEventListener('click', () => setBrushMode(button.dataset.mode));
    });

    byId('brush-slider').addEventListener('input', (event) => {
        AppState.brushWidth = Number(event.target.value);
        byId('brush-size').textContent = event.target.value;
    });

    byId('mask-opacity').addEventListener('input', (event) => {
        byId('canvas-mask').style.opacity = String(Number(event.target.value) / 100);
        byId('opacity-value').textContent = event.target.value;
    });

    bindCompareDrag();
    updateCompare(Number(byId('compare-slider').value));

    byId('btn-undo').addEventListener('click', doUndo);
    byId('btn-redo').addEventListener('click', doRedo);
    byId('btn-clear').addEventListener('click', doClear);
    byId('btn-fill').addEventListener('click', doFillMask);

    document.querySelectorAll('.region-btn').forEach((button) => {
        button.addEventListener('click', () => applyFaceRegionMask(button.dataset.region));
    });

    document.addEventListener('keydown', (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
            event.preventDefault();
            doUndo();
        }
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
            event.preventDefault();
            doRedo();
        }
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
            event.preventDefault();
            saveResult();
        }
    });

    initDragDrop();
    pickRandom();
});
