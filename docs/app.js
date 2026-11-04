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

const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
const comparisons=[...document.querySelectorAll('[data-pair]')].map(row=>{
 const videos=[...row.querySelectorAll('video')],button=row.querySelector('.pair-toggle'),status=row.querySelector('.pair-status');
 const state={row,visible:false,paused:reducedMotion.matches,loaded:false,loading:null,starting:false};
 const wanted=()=>state.visible&&!state.paused&&!document.hidden;
 const update=()=>{button.textContent=state.paused?'Play videos':'Pause videos';button.setAttribute('aria-label',`${state.paused?'Play':'Pause'} both videos: ${row.querySelector('h4').textContent}`);};
 const stop=()=>videos.forEach(v=>v.pause());
 async function load(){
  if(state.loaded)return;
  if(state.loading)return state.loading;
  status.textContent='Loading videos…';
  // Buffer each separate file once, so playback and seeking no longer depend on network delivery.
  state.loading=(async()=>{
   const blobs=await Promise.all(videos.map(async v=>{const r=await fetch(v.dataset.src);if(!r.ok)throw Error('Video unavailable');return r.blob();}));
   await Promise.all(videos.map((v,i)=>new Promise((resolve,reject)=>{
    const cleanup=()=>{v.removeEventListener('canplay',ok);v.removeEventListener('error',fail);};
    const ok=()=>{cleanup();resolve();},fail=()=>{cleanup();reject(Error('Video could not be decoded'));};
    v.addEventListener('canplay',ok);v.addEventListener('error',fail);v.loop=true;v.src=URL.createObjectURL(blobs[i]);v.load();
   })));
   state.loaded=true;status.textContent='';
  })().catch(e=>{state.loading=null;throw e;});
  return state.loading;
 }
 async function play(){
  if(!wanted()||state.starting)return;
  state.starting=true;
  try{
   await load();if(!wanted())return;
   videos[1].currentTime=videos[0].currentTime;
   await Promise.all(videos.map(v=>v.play()));
   if(!wanted())stop();
  }catch(e){stop();state.paused=true;update();status.textContent='Could not load videos. Press Play videos to retry.';}
  finally{state.starting=false;}
 }
 // Use LightInteraction's small drift correction without interrupting both players.
 function sync(){
  if(state.loaded&&wanted()&&!videos.some(v=>v.paused||v.seeking)&&Math.abs(videos[1].currentTime-videos[0].currentTime)>.06)videos[1].currentTime=videos[0].currentTime;
  requestAnimationFrame(sync);
 }
 requestAnimationFrame(sync);
 button.addEventListener('click',()=>{state.paused=!state.paused;update();state.paused?stop():play();});
 state.play=play;state.stop=stop;update();return state;
});
const rowsObserver=new IntersectionObserver(entries=>entries.forEach(e=>{const s=comparisons.find(s=>s.row===e.target);s.visible=e.isIntersecting;s.visible?s.play():s.stop();}),{threshold:.1});
comparisons.forEach(s=>rowsObserver.observe(s.row));
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
