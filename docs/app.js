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
const comparisons = [...document.querySelectorAll('[data-pair]')].map(row => {
  const videos=[...row.querySelectorAll('video')],button=row.querySelector('.pair-toggle'),status=row.querySelector('.pair-status');
  const state={row,videos,visible:false,paused:reducedMotion.matches,token:0,starting:false};
  const shouldPlay=()=>state.visible && !state.paused && !document.hidden;
  const update=()=>{button.textContent=state.paused?'Play videos':'Pause videos';button.setAttribute('aria-label',`${state.paused?'Play':'Pause'} both videos: ${row.querySelector('h4').textContent}`);};
  const stop=()=>{state.token++;videos.forEach(v=>{v.pause();v.playbackRate=1;});};
  function ready(v){
    if(v.error)return Promise.reject(v.error);
    if(v.readyState>=3)return Promise.resolve();
    return new Promise((resolve,reject)=>{
      const clean=()=>{v.removeEventListener('canplay',ok);v.removeEventListener('error',fail);};
      const ok=()=>{clean();resolve();},fail=()=>{clean();reject(v.error);};
      v.addEventListener('canplay',ok);v.addEventListener('error',fail);
    });
  }
  async function play(){
    if(!shouldPlay() || state.starting)return;
    state.starting=true;const token=++state.token;
    try{
      videos.forEach(v=>{v.pause();v.playbackRate=1;if(!v.src){v.preload='auto';v.src=v.dataset.src;v.load();}});
      await Promise.all(videos.map(ready));
      if(token!==state.token || !shouldPlay())return;
      const time=Math.min(...videos.map(v=>v.currentTime));
      await Promise.all(videos.map(v=>new Promise(resolve=>{
        if(!v.seeking && Math.abs(v.currentTime-time)<.02){resolve();return;}
        v.addEventListener('seeked',resolve,{once:true});v.currentTime=time;
      })));
      await Promise.all(videos.map(ready));
      if(token!==state.token || !shouldPlay())return;
      await Promise.all(videos.map(v=>v.play()));
      if(!shouldPlay())videos.forEach(v=>v.pause());
      status.textContent='';
    }catch(e){
      if(token===state.token){stop();state.paused=true;update();status.textContent='Press Play videos to retry.';}
    }finally{
      state.starting=false;
      if(token!==state.token && shouldPlay())play();
    }
  }
  // Small clock drift is corrected gradually; avoid repeated pause/seek cycles.
  const timer=setInterval(()=>{
    if(!shouldPlay() || state.starting || videos.some(v=>v.paused || v.seeking))return;
    if(videos.some(v=>v.readyState<3)){stop();play();return;}
    const drift=videos[0].currentTime-videos[1].currentTime;
    if(Math.abs(drift)>.18){stop();play();}
    else videos[1].playbackRate=Math.abs(drift)<.015?1:drift>0?1.04:.96;
  },100);
  videos.forEach(v=>{
    v.addEventListener('waiting',()=>{if(shouldPlay() && !state.starting && !v.seeking){stop();play();}});
    v.addEventListener('ended',()=>{stop();videos.forEach(x=>x.currentTime=0);play();});
  });
  button.addEventListener('click',()=>{state.paused=!state.paused;update();state.paused?stop():play();});
  state.play=play;state.stop=stop;update();return state;
});
const comparisonObserver=new IntersectionObserver(entries=>{
  entries.forEach(entry=>{
    const s=comparisons.find(x=>x.row===entry.target);s.visible=entry.isIntersecting;s.visible?s.play():s.stop();
  });
},{threshold:.15});
comparisons.forEach(s=>comparisonObserver.observe(s.row));
document.addEventListener('visibilitychange',()=>comparisons.forEach(s=>document.hidden?s.stop():s.visible&&s.play()));

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
