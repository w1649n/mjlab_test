import sys, json, math
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/home/robotserver3/robert/mjlab_test')
sys.path.insert(0, str(ROOT/'scripts'))
from export_rl_mpc_policy import load_actor
from validate_rl_mpc import _place_all_terrain_types_at_level, _set_exact_manual_command
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.rl_mpc.config.g23.env_cfgs import syncai_g23_pure_mpc_validation_env_cfg

torch.set_num_threads(1)
run=ROOT/'logs/rsl_rl/g23_rl_mpc_foot_state_history_v3/2026-09-09_15-00-51'
cfg=syncai_g23_pure_mpc_validation_env_cfg()
cfg.episode_length_s=30.0
cfg.auto_reset=False
env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
out=[]
try:
 for name,level in [('bare',0),('model_3000.pt',0),('model_9999.pt',0),('model_9999.pt',3)]:
  torch.manual_seed(42);np.random.seed(42)
  actor=None if name=='bare' else load_actor(run/name)[0].as_onnx(verbose=False).eval()
  _place_all_terrain_types_at_level(env,level)
  env.reset()
  term=env.action_manager.get_term('foot_placement')
  robot=env.scene['robot']
  cmds=env.command_manager.get_term('twist')
  start=robot.data.root_link_pos_w.clone()
  results=[]
  for phase,duration,command in [('stand',3,(0.,0.,0.)),('walk',10,(0.2,0.,0.)),('stop',7,(0.,0.,0.))]:
   for e in range(env.num_envs):_set_exact_manual_command(cmds,e,command)
   # A manual command changes between policy steps. Update only the newest
   # frame's command so old frames retain the commands that actually occurred.
   records=[]
   for step in range(round(duration/env.step_dt)):
    obs=term.state_history.clone()
    yaw=obs[:,2]
    obs[:,61]=yaw.cos()*command[0]-yaw.sin()*command[1]
    obs[:,62]=yaw.sin()*command[0]+yaw.cos()*command[1]
    obs[:,63]=command[2]
    term._observation_history.frames[:,0,61:64]=obs[:,61:64]
    with torch.inference_mode():
     raw=torch.zeros((env.num_envs,8)) if actor is None else actor(obs)
    _,_,done,truncated,_=env.step(raw.clamp(-1,1))
    data=robot.data
    contacts=env.scene['feet_ground_contact'].data.found.reshape(env.num_envs,4,-1).gt(0).any(-1)
    records.append({
     'velocity':torch.cat((data.root_link_lin_vel_b[:,:2],data.root_link_ang_vel_b[:,2:3]),dim=1).tolist(),
     'pose_mse':((data.joint_pos-data.default_joint_pos)**2).mean(-1).tolist(),
     'all_contact':contacts.all(-1).float().tolist(),
     'mode':term.gait_modes.tolist(),
     'clip':(raw.abs()>1).float().mean(-1).tolist(),
     'action':raw.clamp(-1,1).tolist(),
    })
    if bool(done.any() or truncated.any()):break
   tail=records[len(records)//2:]
   vel=np.array([r['velocity'] for r in tail]);acts=np.array([r['action'] for r in tail])
   results.append({'phase':phase,'seconds':len(records)*env.step_dt,'terminated':done.tolist(),
    'mean_velocity':vel.mean(0).tolist(),
    'tracking_mae':np.abs(vel-np.array(command)).mean(0).tolist(),
    'pose_rmse_rad':np.sqrt(np.array([r['pose_mse'] for r in tail]).mean(0)).tolist(),
    'four_contact_fraction':np.array([r['all_contact'] for r in tail]).mean(0).tolist(),
    'deterministic_clip_fraction':np.array([r['clip'] for r in tail]).mean(0).tolist(),
    'mean_action':acts.mean((0,1)).tolist(),
    'final_modes':term.gait_modes.tolist(),
    'displacement_xyz':(robot.data.root_link_pos_w-start).tolist()})
   if bool(done.any() or truncated.any()):break
  out.append({'model':name,'level':level,'phases':results})
  (ROOT/'analysis/rlmpc_2026-09-10/deterministic_eval.json').write_text(json.dumps(out,indent=2))
  print('EVAL_RESULT',json.dumps(out[-1]),flush=True)
finally:
 env.close()
