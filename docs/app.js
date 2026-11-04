'use strict';

const models = {
  hy: {name: 'HY-WorldPlay', storage: [20.96, 1.55, .80, 1.02], psnr: ['—', '18.56', '19.56', '19.04'], compression: [1, 13.50, 26.09, 20.53], speedup: [1, .92, .96, 1.06], attention: '37.93'},
  lingbot: {name: 'LingBot-World 2.0', storage: [20.71, 4.73, 1.46, 1.78], psnr: ['—', '20.10', '20.81', '20.68'], compression: [1, 4.38, 14.18, 11.62], speedup: [1, .79, .93, .98], attention: '34.32'},
  mg: {name: 'Matrix-Game 3.5', storage: [11.48, 1.24, .80, .96], psnr: ['—', '18.83', '19.08', '18.79'], compression: [1, 9.28, 14.26, 11.94], speedup: [1, .57, .92, 1.04], attention: '20.83'}
};
const $ = id => document.getElementById(id);
const methods = ['Original', 'KVTC', 'NanoKV', 'NanoKV-Fast'];
function showResults(model) {
  const d = models[model];
  $('results-caption').textContent = `${d.name} · main-table results`;
  $('storage-bars').innerHTML = methods.map((name, i) => `<div class="bar-row ${i === 2 ? 'ours' : i === 3 ? 'fast' : ''}"><span>${name}</span><div class="bar-track"><div class="bar-fill" style="width:${100*d.storage[i]/d.storage[0]}%"></div></div><b>${d.storage[i].toFixed(2)}</b></div>`).join('');
  $('results-body').innerHTML = methods.map((name, i) => `<tr class="${i === 2 ? 'highlight' : i === 3 ? 'fast-row' : ''}"><td>${name}</td><td>${d.psnr[i]}</td><td>${d.compression[i].toFixed(2)}×</td><td>${d.speedup[i].toFixed(2)}×</td></tr>`).join('');
  $('attention-saving').textContent = `${d.attention}% less historical-attention computation`;
}
$('results-model').addEventListener('change', e => showResults(e.target.value));
showResults('hy');

// Ambient previews load only when visible, and honor reduced-motion preferences.
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let ambientPaused = reducedMotion.matches;
const previews = [...document.querySelectorAll('[data-preview]')];
function updateAmbientButton() {
  $('ambient-toggle').textContent = ambientPaused ? 'Play previews' : 'Pause previews';
  $('ambient-toggle').setAttribute('aria-pressed', String(ambientPaused));
}
function startPreview(video) {
  if (!video.src) video.src = video.dataset.preview;
  video.play().catch(() => {});
}
const observer = new IntersectionObserver(entries => {
  entries.forEach(({target, isIntersecting}) => {
    target.dataset.visible = String(isIntersecting);
    if (isIntersecting && !ambientPaused && !document.hidden) startPreview(target);
    else target.pause();
  });
}, {threshold: .1});
previews.forEach(v => observer.observe(v));
$('ambient-toggle').addEventListener('click', () => {
  ambientPaused = !ambientPaused;
  previews.forEach(v => ambientPaused ? v.pause() : v.dataset.visible === 'true' && startPreview(v));
  updateAmbientButton();
});
updateAmbientButton();

const original = $('original-video');
const candidate = $('candidate-video');
const pair = [original, candidate];
let scenes = [], currentModel = 'hy', currentScene, mode = 'nanokv', loading = false;
let playing = false, generation = 0, seeking = false;
const clock = time => `${Math.floor(Math.max(0,time)/60)}:${String(Math.floor(Math.max(0,time)%60)).padStart(2,'0')}`;
function pausePair() {
  playing = false;
  pair.forEach(v => v.pause());
  $('play-pair').textContent = '▶ Play comparison';
  $('play-pair').setAttribute('aria-label', 'Play both comparison videos');
}
async function playPair() {
  if (loading) return;
  if (original.currentTime >= original.duration - .1) pair.forEach(v => v.currentTime = 0);
  candidate.currentTime = original.currentTime;
  try {
    await Promise.all(pair.map(v => v.play()));
    playing = true;
    $('play-pair').textContent = 'Ⅱ Pause comparison';
    $('play-pair').setAttribute('aria-label', 'Pause both comparison videos');
    $('video-status').textContent = '';
  } catch {
    pausePair();
    $('video-status').textContent = 'Playback could not start. Press Play comparison to try again.';
  }
}
function tick() {
  const duration = Math.min(original.duration || 0, candidate.duration || 0);
  if (duration > 0 && Number.isFinite(duration)) {
    if (!seeking) $('timeline').value = original.currentTime;
    $('video-time').textContent = `${clock(original.currentTime)} / ${clock(duration)}`;
    if (playing && !seeking && !pair.some(v => v.seeking) && Math.abs(original.currentTime - candidate.currentTime) > .12) candidate.currentTime = original.currentTime;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
$('play-pair').addEventListener('click', () => playing ? pausePair() : playPair());
$('restart-pair').addEventListener('click', () => {
  if (loading) return;
  pair.forEach(v => v.currentTime = 0);
});
$('timeline').addEventListener('input', e => {
  seeking = true;
  const time = Number(e.target.value);
  pausePair();
  pair.forEach(v => v.currentTime = time);
  seeking = pair.some(v => v.seeking);
});
pair.forEach(v => v.addEventListener('seeked', () => {
  if (!pair.some(video => video.seeking)) seeking = false;
}));
pair.forEach(v => v.addEventListener('ended', pausePair));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {pausePair(); previews.forEach(v => v.pause());}
  else if (!ambientPaused) previews.filter(v => v.dataset.visible === 'true').forEach(startPreview);
});
// Pause comparisons as they leave the viewport, without resuming unsolicited playback.
new IntersectionObserver(entries => {
  if (!entries[0].isIntersecting) pausePair();
}, {threshold: .05}).observe($('compare-stage'));

function selectScene(scene) {
  generation += 1;
  const token = generation;
  pausePair();
  loading = true;
  currentScene = scene;
  $('play-pair').disabled = true;
  $('timeline').disabled = true;
  $('timeline').value = 0;
  $('video-time').textContent = '0:00 / 0:00';
  $('video-status').textContent = 'Loading selected scene…';
  const d = models[currentModel], index = mode === 'fast' ? 3 : 2;
  $('scene-title').textContent = scene.title;
  const actions = {left_right: 'Camera pans left, then right', right_left: 'Camera pans right, then left', forward_backward: 'Camera moves forward, then backward'};
  $('scene-meta').textContent = `${scene.source_kind === 'teaser' ? 'Paper teaser' : 'Paper appendix'} · ${actions[scene.action] || scene.action}`;
  $('scene-compression').textContent = `${d.compression[index].toFixed(2)}×`;
  $('original-storage').textContent = `${d.storage[0].toFixed(2)} GiB KV`;
  $('candidate-storage').textContent = `${d.storage[index].toFixed(2)} GiB KV`;
  $('candidate-label').innerHTML = `<i class="legend ours"></i>${methods[index]}`;
  candidate.setAttribute('aria-label', `${methods[index]} video: ${scene.title}`);
  original.setAttribute('aria-label', `Original video: ${scene.title}`);
  document.querySelectorAll('[data-scene]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.scene === scene.id)));
  let ready = 0;
  pair.forEach((video, i) => {
    video.onloadeddata = () => {
      if (token !== generation) return;
      ready += 1;
      if (ready === 2) {
        loading = false;
        $('play-pair').disabled = false;
        $('timeline').disabled = false;
        $('timeline').max = Math.min(original.duration, candidate.duration);
        $('video-status').textContent = '';
      }
    };
    video.onerror = () => {
      if (token === generation) $('video-status').textContent = 'This video could not be loaded. Choose another scene or reload the page.';
    };
    video.poster = scene.poster;
    video.src = scene.videos[i === 0 ? 'original' : mode].src;
    video.load();
  });
}
function selectModel(model) {
  currentModel = model;
  document.querySelectorAll('[data-model]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.model === model)));
  const available = scenes.filter(s => s.model === model);
  $('scene-buttons').replaceChildren(...available.map(scene => {
    const b = document.createElement('button');
    b.type = 'button'; b.dataset.scene = scene.id;
    b.setAttribute('aria-pressed', 'false');
    const img = document.createElement('img'); img.src = scene.poster; img.alt = ''; img.width = 48; img.height = 32;
    b.append(img, document.createTextNode(scene.title));
    b.addEventListener('click', () => selectScene(scene));
    return b;
  }));
  $('results-model').value = model;
  showResults(model);
  selectScene(available[0]);
}
document.querySelectorAll('[data-model]').forEach(b => b.addEventListener('click', () => {if (scenes.length) selectModel(b.dataset.model);}));
document.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => {
  mode = b.dataset.mode;
  document.querySelectorAll('[data-mode]').forEach(btn => btn.setAttribute('aria-pressed', String(btn === b)));
  if (currentScene) selectScene(currentScene);
}));
fetch('assets/scenes.json').then(r => {if (!r.ok) throw new Error('Scene manifest unavailable'); return r.json();}).then(data => {
  scenes = data.scenes;
  selectModel(currentModel);
}).catch(() => {$('video-status').textContent = 'Scene information could not be loaded. Please reload the page.';});

$('copy-citation').addEventListener('click', async () => {
  const text = $('citation-text').textContent;
  try {
    await navigator.clipboard.writeText(text);
    $('copy-citation').textContent = 'Copied';
    $('copy-status').textContent = 'Citation copied to clipboard.';
    setTimeout(() => {$('copy-citation').textContent = 'Copy BibTeX';}, 2000);
  } catch {
    const selection = window.getSelection(), range = document.createRange();
    range.selectNodeContents($('citation-text')); selection.removeAllRanges(); selection.addRange(range);
    $('copy-status').textContent = 'Citation selected. Use your keyboard to copy it.';
  }
});
