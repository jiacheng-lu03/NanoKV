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

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

// Each visible row plays its two videos together, following the showcase layout.
const comparisons = [...document.querySelectorAll('[data-pair]')].map(row => {
  const videos = [...row.querySelectorAll('video')];
  const button = row.querySelector('.pair-toggle');
  const status = row.querySelector('.pair-status');
  const state = {row, videos, visible:false, paused:reducedMotion.matches, token:0};
  function update() {
    button.textContent = state.paused ? 'Play videos' : 'Pause videos';
    button.setAttribute('aria-label', `${state.paused ? 'Play' : 'Pause'} both videos: ${row.querySelector('h4').textContent}`);
  }
  function stop() { state.token++; videos.forEach(v => v.pause()); }
  async function play() {
    if (state.paused || !state.visible || document.hidden) return;
    const token = ++state.token;
    videos.forEach(v => { if (!v.src) {v.src = v.dataset.src; v.load();} });
    try {
      await Promise.all(videos.map(v => v.readyState >= 3 ? Promise.resolve() : new Promise((resolve, reject) => {
        v.addEventListener('canplay', resolve, {once:true});
        v.addEventListener('error', reject, {once:true});
      })));
      if (token !== state.token || state.paused || !state.visible || document.hidden) return;
      // Seek both paused videos before releasing them together.
      const time = Math.min(...videos.map(v=>v.currentTime));
      await Promise.all(videos.map(v=>new Promise(resolve=>{
        if(Math.abs(v.currentTime-time)<.001 && !v.seeking){resolve();return;}
        v.addEventListener('seeked',resolve,{once:true});v.currentTime=time;
      })));
      if(token!==state.token || state.paused || !state.visible || document.hidden)return;
      await Promise.all(videos.map(v => v.play()));
      if (state.paused || !state.visible || document.hidden) videos.forEach(v => v.pause());
      status.textContent = '';
    } catch {
      if (token !== state.token) return;
      stop(); state.paused=true; update();
      status.textContent='Press Play videos to retry playback.';
    }
  }
  function syncFrames(){
    const end=Math.min(...videos.map(v=>v.duration));
    if(!state.paused && state.visible && !videos.some(v=>v.paused || v.seeking) && Number.isFinite(end) && Math.max(...videos.map(v=>v.currentTime))>=end-.08){
      stop();videos.forEach(v=>v.currentTime=0);play();
    }
    if(!state.paused && state.visible && !videos.some(v=>v.paused) &&
      (videos.some(v=>v.readyState<3 || v.seeking) || Math.abs(videos[0].currentTime-videos[1].currentTime)>1/30)){
      stop();play();
    }
    requestAnimationFrame(syncFrames);
  }
  requestAnimationFrame(syncFrames);
  videos.forEach(v => {
    v.addEventListener('ended', () => {stop();videos.forEach(video=>video.currentTime=0);play();});
  });
  button.addEventListener('click', () => {state.paused=!state.paused;update();state.paused?stop():play();});
  state.play=play;state.stop=stop;update();return state;
});
const comparisonObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    const state=comparisons.find(s=>s.row===entry.target);
    state.visible=entry.isIntersecting;
    state.visible?state.play():state.stop();
  });
}, {threshold:.05});
comparisons.forEach(s=>comparisonObserver.observe(s.row));
document.addEventListener('visibilitychange',()=>{
  if(document.hidden){comparisons.forEach(s=>s.stop());}
  else{comparisons.filter(s=>s.visible).forEach(s=>s.play());}
});

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
