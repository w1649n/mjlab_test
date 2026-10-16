"""Aggregate evaluate_goals.py outputs into one KPI row per goal."""

import json
import sys

import numpy as np


def pct(x):
  return f"{100 * x:5.1f}%"


def kpis(path):
  res = json.loads(open(path).read())
  rows = res["rows"]
  flat = [r for r in rows if r["kind"] == "flat"]
  stairs = [r for r in rows if r["kind"] != "flat"]
  k = {}
  k["G1 flat walk survival"] = pct(np.mean([r["walk_survived"] for r in flat]))
  vx = [r for r in flat if r["walk_survived"] and r["cmd"][1] == 0 and r["cmd"][2] == 0]
  k["G1 flat |vx err| mean m/s"] = f"{np.mean([r['track_mae'][0] for r in vx]):.3f}"
  stop = [r["stop_4_7s"] for r in flat if "stop_4_7s" in r]
  rm = np.array([s["joint_rmse_deg"] for s in stop])
  mx = np.array([s["joint_max_abs_deg"] for s in stop])
  fc = np.array([s["four_contact_frac"] for s in stop])
  sp = np.array([s["base_speed"] for s in stop])
  k["G2 stop pass (rmse<=3,max<=6,4feet)"] = pct(
    np.mean((rm <= 3) & (mx <= 6) & (fc >= 0.95) & (sp <= 0.05))
  )
  k["G2 stop joint rmse p50/p90 deg"] = (
    f"{np.median(rm):.1f}/{np.percentile(rm, 90):.1f}"
  )
  k["G2 stop joint max-abs p90 deg"] = f"{np.percentile(mx, 90):.1f}"
  straight = [r for r in vx]
  dm = np.array([r["heading_drift_max_deg"] for r in straight])
  k["G3 flat straight |yaw drift|<=2deg"] = pct(np.mean(dm <= 2))
  k["G3 flat drift p50/p90 deg (8s)"] = (
    f"{np.median(dm):.1f}/{np.percentile(dm, 90):.1f}"
  )
  for lo, hi, name in ((0.0, 0.16, "<=15cm"), (0.17, 0.19, "18cm")):
    for kind in ("up", "down"):
      rs = [r for r in stairs if r["kind"] == kind and lo <= r["height"] <= hi]
      k[f"G4 {kind} {name} success"] = pct(np.mean([r["success"] for r in rs]))
  for kind in ("up", "down"):
    for vx_ in (0.3, 0.5, 0.8, 1.0):
      rs = [
        r
        for r in stairs
        if r["kind"] == kind and abs(r["height"] - 0.20) < 1e-6 and r["cmd"][0] == vx_
      ]
      k[f"G6 {kind} 20cm v{vx_} success"] = pct(np.mean([r["success"] for r in rs]))
  for kind in ("up", "down"):
    rs = [r for r in stairs if r["kind"] == kind and r["success"]]
    rs = [r for r in rs if "heading_drift_max_deg" in r]
    dm = np.array([r["heading_drift_max_deg"] for r in rs])
    k[f"G5 {kind} |yaw drift|<=2deg"] = pct(np.mean(dm <= 2))
    k[f"G5 {kind} drift p50/p90 deg"] = (
      f"{np.median(dm):.1f}/{np.percentile(dm, 90):.1f}"
    )
  up = [
    r
    for r in stairs
    if r["kind"] == "up"
    and r["success"]
    and "nose_up_mean_deg" in r
    and r["height"] >= 0.15
  ]
  down = [
    r
    for r in stairs
    if r["kind"] == "down"
    and r["success"]
    and "nose_up_mean_deg" in r
    and r["height"] >= 0.15
  ]
  k["G7 up>=15cm mean nose-up deg"] = (
    f"{np.mean([r['nose_up_mean_deg'] for r in up]):+.1f}"
  )
  k["G7 up>=15cm never nose-down"] = pct(
    np.mean([r["nose_up_min_deg"] >= 0 for r in up])
  )
  k["G7 down>=15cm worst nose-down deg"] = (
    f"{-np.min([r['nose_up_min_deg'] for r in down]):.1f}"
  )
  k["G7 down>=15cm nose-down<=45"] = pct(
    np.mean([r["nose_up_min_deg"] >= -45 for r in down])
  )
  k["G7 stairs worst |roll| deg"] = (
    f"{np.max([r['roll_abs_max_deg'] for r in up + down]):.1f}"
  )
  names = res["actuator_names"]
  for kind in ("up", "down"):
    rs = [
      r
      for r in stairs
      if r["kind"] == kind
      and r["success"]
      and "tau_rms_ratio" in r
      and abs(r["height"] - 0.20) < 1e-6
    ]
    ratio = np.array([r["tau_rms_ratio"] for r in rs])
    worst = ratio.max(1)
    j = int(np.argmax(ratio.mean(0)))
    k[f"G8 {kind} 20cm RMS/rated hottest"] = f"{ratio.mean(0)[j]:.2f} ({names[j]})"
    k[f"G8 {kind} 20cm runs <=1.0 rated"] = pct(np.mean(worst <= 1.0))
  rs = [r for r in stairs if r["success"] and "tau_peak_ratio" in r]
  k["G8 stairs peak/J60 peak max"] = (
    f"{np.max([max(r['tau_peak_ratio']) for r in rs]):.2f}"
  )
  return k


paths = sys.argv[1:]
tables = [kpis(p) for p in paths]
labels = [p.split("/")[-1].replace(".json", "") for p in paths]
width = max(len(x) for x in labels) + 2
print(f"{'KPI':40s}" + "".join(f"{lab:>{width}s}" for lab in labels))
for key in tables[0]:
  print(f"{key:40s}" + "".join(f"{t[key]:>{width}s}" for t in tables))
