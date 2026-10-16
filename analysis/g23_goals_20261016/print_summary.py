"""Print compact goal tables from evaluate_goals.py outputs."""

import json
import sys

for path in sys.argv[1:]:
  res = json.loads(open(path).read())
  s = res["summary"]
  print(f"\n##### {path}  actuator={res['actuator']} dr={res['domain_rand']}")
  print(
    "FLAT  cmd              walk_ok  trackMAE(vx,vy,wz)   hdg_max p50/p90/max  "
    "<=2deg | stop rmse p50/p90/max  maxabs p90  4contact  pass"
  )
  for cmd, v in s["flat"].items():
    hd = v.get("heading_drift_max_deg_p50_p90_max")
    hd_s = "/".join(f"{x:.1f}" for x in hd) if hd else "-"
    w2 = f"{v['heading_within_2deg']:.2f}" if "heading_within_2deg" in v else "-"
    rm = v.get("stop_joint_rmse_deg_p50_p90_max")
    rm_s = "/".join(f"{x:.1f}" for x in rm) if rm else "-"
    mx = v.get("stop_joint_maxabs_deg_p50_p90_max")
    print(
      f"  {cmd:18s} {v['walk_survival']:.2f}  {str(v.get('track_mae_vx_vy_wz')):22s}"
      f" {hd_s:16s} {w2:5s} | {rm_s:16s} {mx[1] if mx else 0:5.1f}"
      f"  {v.get('stop_four_contact_mean', 0):.2f}  {v.get('stop_pass_rate', 0):.2f}"
    )
  print(
    "STAIRS cond            succ  fails                     hdg p50/p90/max <=2  "
    "noseUp mean/min/max  roll  tauRMS/rated(max-act)  peak  speed"
  )
  for k, v in s["stairs"].items():
    hd = v.get("heading_drift_max_deg_p50_p90_max")
    hd_s = "/".join(f"{x:.1f}" for x in hd) if hd else "-"
    nu = (
      f"{v['nose_up_mean_deg']:+.1f}/{v['nose_up_min_deg_worst']:+.1f}"
      f"/{v['nose_up_max_deg_best']:+.1f}"
      if "nose_up_mean_deg" in v
      else "-"
    )
    print(
      f"  {k:20s} {v['success_rate']:.2f}  {str(v['fail_reasons'])[:24]:24s}"
      f"  {hd_s:15s} {v.get('heading_within_2deg', float('nan')):.2f}  {nu:20s}"
      f" {v.get('roll_abs_max_deg_worst', float('nan')):5.1f}"
      f"  {v.get('tau_rms_ratio_max_actuator_mean', float('nan')):5.2f}"
      f"  {v.get('tau_peak_ratio_max', float('nan')):5.2f}"
      f"  {v.get('mean_speed', float('nan')):.2f}"
    )
