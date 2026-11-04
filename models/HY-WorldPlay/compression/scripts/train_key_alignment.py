"""Serial online Key block-output alignment with a compressed layer prefix."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True,help='JSON list: id, image, prompt, pose')
    p.add_argument('--basis-root',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--action-ckpt',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--rank',type=int,choices=[256,512],default=256)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    cases=json.loads(a.manifest.read_text())
    if not isinstance(cases,list) or not cases:p.error('Manifest must be a nonempty list')
    for case in cases:
        if 'actions' not in case and 'pose' in case:
            case['actions'] = case['pose']
        for key in ['id','image','prompt','actions']:
            if key not in case:p.error(f'Missing case field: {key}')
    out=a.output_root.resolve(); init=out/'initial'
    if out.exists() and not a.resume:
        p.error('Output already exists; use a new directory or --resume')
    init_cmd=[sys.executable,str(ROOT/'scripts/init_key.py'),
              '--basis-root',str(a.basis_root.resolve()),'--output-root',str(init),
              '--layers','all','--rank',str(a.rank),'--semantic-smooth',
              '--semantic-clusters','64','--semantic-iters','2']
    if a.dry_run:print('Initialize:',init_cmd)
    elif not (a.resume and (init/'manifest.json').is_file()):
        subprocess.run(init_cmd,cwd=ROOT,check=True)
    for begin in range(0,54,9):
        layers=','.join(map(str,range(begin,begin+9)))
        prefix=','.join(map(str,range(begin+9)))
        for case in cases:
            env=os.environ.copy()
            for key in list(env):
                if key.startswith('HY_'): del env[key]
            env.update({
              'PYTHONPATH':str(ROOT),
              'HY_PRE_POSITION_VISION_KV_CACHE':'1',
              'HY_KV_CODEC_ENABLE':'0',
              'HY_KV_CODEC_SEMANTIC_CLUSTERS':'64','HY_KV_CODEC_SEMANTIC_ITERS':'2',
              'HY_KV_CORRUPTION_K_ENABLE':'1','HY_KV_CORRUPTION_K_MODE':'dcae',
              'HY_KV_CORRUPTION_K_START_CHUNK':'0',
              'HY_KV_CORRUPTION_K_LAYERS':prefix,
              'HY_KV_CORRUPTION_K_DCAE_CKPT_DIR':str(init),
              'HY_KV_CORRUPTION_K_DCAE_MODEL_VARIANT':'global_pca_linear_d512',
              'HY_KV_CORRUPTION_K_GAMMA':'1.0',
              'HY_KV_CORRUPTION_K_SEMANTIC_SMOOTH':'1','HY_KV_CORRUPTION_K_FORCE_FP32_LINEAR':'1',
              'HY_KV_CORRUPTION_K_TRAINABLE':'1','HY_KV_CORRUPTION_K_TRAINABLE_LAYERS':layers,
              'HY_KV_CORRUPTION_K_DECODER_ONLY':'1','HY_KV_CORRUPTION_K_FP32_MASTER':'1',
              'HY_KV_CORRUPTION_K_LATENT_INT4_TOKEN':'1','HY_KV_CORRUPTION_K_LR':'5e-6',
              'HY_KV_CORRUPTION_K_WEIGHT_DECAY':'0','HY_KV_CORRUPTION_K_OPTIMIZER':'adamw',
              'HY_KV_CORRUPTION_K_GRAD_CLIP':'1',
              'HY_FUNCTIONAL_BLOCK_FT_ENABLE':'1','HY_FUNCTIONAL_BLOCK_FT_TARGET':'k',
              'HY_FUNCTIONAL_BLOCK_FT_LOSS_MODE':'block','HY_FUNCTIONAL_BLOCK_FT_LAYERS':layers,
              'HY_FUNCTIONAL_BLOCK_FT_LOCAL_TARGET':'hidden','HY_FUNCTIONAL_BLOCK_FT_LOCAL_WEIGHT':'1',
              'HY_FUNCTIONAL_BLOCK_FT_PROP1_ENABLE':'0','HY_FUNCTIONAL_BLOCK_FT_PROP1_WEIGHT':'0',
              'HY_FUNCTIONAL_BLOCK_FT_ONLY_STEP':'3','HY_FUNCTIONAL_BLOCK_FT_STEPS_PER_CHUNK':'1',
              'HY_FUNCTIONAL_BLOCK_FT_CHUNKS':'1,2,3,4','HY_FUNCTIONAL_BLOCK_FT_INNER_STEPS':'5',
              'HY_FUNCTIONAL_BLOCK_FT_TEACHER_FORCED':'1','HY_FUNCTIONAL_BLOCK_FT_TEACHER_FORCE_ALL_STEPS':'0',
              'HY_FUNCTIONAL_BLOCK_FT_TEACHER_CACHE_MODE':'same_state_group',
              'HY_FUNCTIONAL_BLOCK_FT_PREFIX_REUSE_STRICT':'1',
              'HY_FUNCTIONAL_BLOCK_FT_RECON_WEIGHT':'0.2','HY_FUNCTIONAL_BLOCK_FT_COS_WEIGHT':'0.02',
              'HY_FUNCTIONAL_BLOCK_FT_ANCHOR_SAMPLES':'65536',
              'HY_FUNCTIONAL_BLOCK_FT_TOKEN_SAMPLES':'256','HY_FUNCTIONAL_BLOCK_FT_CHANNEL_SAMPLES':'256',
              'HY_FUNCTIONAL_BLOCK_FT_OUT_ROOT':str(out),'HY_K_FUNCTIONAL_FT_CKPT_DIR':str(out),
              'HY_FUNCTIONAL_ACCUM_ATOM_ID_EXACT':'1',
              'HY_FUNCTIONAL_ACCUM_ATOM_PREFIX':f'g{begin//9}_{case["id"]}',
              'HY_STOP_AFTER_AR_CHUNK':'4',
            })
            cmd=[sys.executable,str(ROOT/'hyvideo/generate.py'),
                 '--prompt',case['prompt'],'--image_path',str(Path(case['image']).resolve()),
                 '--pose',case['actions'],'--model_path',str(a.model_path.resolve()),
                 '--action_ckpt',str(a.action_ckpt.resolve()),'--resolution','480p',
                 '--aspect_ratio','16:9','--video_length','253','--seed','1',
                 '--rewrite','false','--sr','false','--few_step','true',
                 '--num_inference_steps','4','--model_type','ar',
                 '--use_vae_parallel','false','--use_sageattn','false','--use_fp8_gemm','false',
                 '--transformer_resident_ar_rollout','false','--offloading','true',
                 '--group_offloading','false','--output_path',str(out/'videos'/f'g{begin//9}'/case['id'])]
            if a.dry_run: print('Fit:',cmd)
            else: subprocess.run(cmd,env=env,cwd=ROOT,check=True)

if __name__=='__main__':main()
