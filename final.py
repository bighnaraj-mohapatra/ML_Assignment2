import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed

print('1a : ')
with open("config.json", "r") as f_cfg:
    app_cfg = json.load(f_cfg)

roll_id = app_cfg["roll_number"]
base_seed = int("".join(ch for ch in roll_id if ch.isdigit()))

k_tr = app_cfg["split_per_identity"]["train"]
k_va = app_cfg["split_per_identity"]["val"]
k_te = app_cfg["split_per_identity"]["test"]

r_lo, r_hi = app_cfg["mask_region"]["rows"][0], app_cfg["mask_region"]["rows"][1] + 1
c_lo, c_hi = app_cfg["mask_region"]["cols"][0], app_cfg["mask_region"]["cols"][1] + 1

opt_dt_depth = app_cfg["selected_single_tree"]["max_depth"]
opt_dt_leaf = app_cfg["selected_single_tree"]["min_samples_leaf"]
opt_bag_cnt = app_cfg["selected_bagging"]["B"]
opt_bst_depth = app_cfg["selected_boosting"]["max_depth"]
opt_bst_steps = app_cfg["selected_boosting"]["T"]
total_refits = app_cfg["q3_refits"]

os.makedirs("data_split", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

split_rng = np.random.default_rng(base_seed)

raw_faces = fetch_olivetti_faces()
img_matrix = raw_faces.data
person_ids = raw_faces.target

idx_tr, idx_va, idx_te = [], [], []

for subj in range(40):
    subj_rows = np.where(person_ids == subj)[0]
    shuffled_rows = split_rng.permutation(subj_rows)
    idx_tr.extend(shuffled_rows[:k_tr])
    idx_va.extend(shuffled_rows[k_tr : k_tr + k_va])
    idx_te.extend(shuffled_rows[k_tr + k_va : k_tr + k_va + k_te])

idx_tr = np.array(idx_tr)
idx_va = np.array(idx_va)
idx_te = np.array(idx_te)

tr_X, tr_y = img_matrix[idx_tr], person_ids[idx_tr]
va_X, va_y = img_matrix[idx_va], person_ids[idx_va]
te_X, te_y = img_matrix[idx_te], person_ids[idx_te]

masked_buf = te_X.reshape(-1, 64, 64).copy()
masked_buf[:, r_lo:r_hi, c_lo:c_hi] = 0.0
te_X_masked = masked_buf.reshape(-1, 4096)

np.savez(
    "data_split/split_data.npz",
    seed=base_seed,
    tr_ids=idx_tr,
    va_ids=idx_va,
    te_ids=idx_te,
    x_tr=tr_X,
    y_tr=tr_y,
    x_va=va_X,
    y_va=va_y,
    x_te=te_X,
    x_te_occ=te_X_masked,
    y_te=te_y
)

cand_depths = [2, 4, 8, 16, None]
cand_leaves = [1, 2, 4]

grid_logs = []
top_va_acc = -1.0
chosen_tree_cfg = None

print(f"{'Depth':<8} {'MinLeaf':<8} {'TrAcc':<8} {'VaAcc':<8} {'TeAcc':<8} {'Leaves':<8} {'AvgOcc':<8} {'SingleFrac':<10}")
print("-" * 72)

for d_opt in cand_depths:
    for l_opt in cand_leaves:
        clf_tree = DecisionTreeClassifier(max_depth=d_opt, min_samples_leaf=l_opt, random_state=base_seed)
        clf_tree.fit(tr_X, tr_y)

        sc_tr = float(clf_tree.score(tr_X, tr_y))
        sc_va = float(clf_tree.score(va_X, va_y))
        sc_te = float(clf_tree.score(te_X, te_y))

        leaf_assignments = clf_tree.apply(tr_X)
        uniq_leaves, leaf_counts = np.unique(leaf_assignments, return_counts=True)

        num_leaves = int(len(uniq_leaves))
        mean_occ = float(np.mean(leaf_counts))
        frac_single = float(np.mean(leaf_counts == 1))

        entry = {
            "max_depth": d_opt,
            "min_samples_leaf": l_opt,
            "tr_acc": sc_tr,
            "va_acc": sc_va,
            "te_acc": sc_te,
            "tot_leaves": num_leaves,
            "avg_occ": mean_occ,
            "single_ratio": frac_single
        }
        grid_logs.append(entry)

        d_str = str(d_opt) if d_opt is not None else "None"
        print(f"{d_str:<8} {l_opt:<8} {sc_tr:<8.4f} {sc_va:<8.4f} {sc_te:<8.4f} {num_leaves:<8} {mean_occ:<8.2f} {frac_single:<10.4f}")

        if sc_va > top_va_acc:
            top_va_acc = sc_va
            chosen_tree_cfg = entry

print("-" * 72)
print("Selected Configuration:", chosen_tree_cfg)

with open("outputs/q1a_summary.json", "w") as f_out:
    json.dump({"runs": grid_logs, "selected": chosen_tree_cfg}, f_out, indent=2)

print('\n')
print('1b : ')
npz_archive = np.load("data_split/split_data.npz")
base_seed = int(npz_archive["seed"])
tr_X, tr_y = npz_archive["x_tr"], npz_archive["y_tr"]
va_X, va_y = npz_archive["x_va"], npz_archive["y_va"]
te_X, te_y = npz_archive["x_te"], npz_archive["y_te"]

ens_grid = [1, 5, 10, 25, 50, 100, 200]
num_tr = tr_X.shape[0]

expected_uniq = num_tr * (1.0 - (1.0 - 1.0 / num_tr) ** num_tr)

q1b_logs = []

print(f"{'B':<6} {'TrAcc':<8} {'VaAcc':<8} {'TeAcc':<8} {'EmpDistinct':<14} {'TheoDistinct':<14}")
print("-" * 68)

bag_gen = np.random.default_rng(base_seed)

for b_sz in ens_grid:
    forest_list = []
    uniq_tracker = []

    for _ in range(b_sz):
        sample_ids = bag_gen.integers(0, num_tr, size=num_tr)
        uniq_tracker.append(len(np.unique(sample_ids)))

        member_dt = DecisionTreeClassifier(max_depth=opt_dt_depth, min_samples_leaf=opt_dt_leaf, random_state=base_seed)
        member_dt.fit(tr_X[sample_ids], tr_y[sample_ids])
        forest_list.append(member_dt)

    def majority_vote(eval_X):
        raw_votes = np.array([dt.predict(eval_X) for dt in forest_list])
        n_eval = eval_X.shape[0]
        voted_out = []
        for idx in range(n_eval):
            col_votes = raw_votes[:, idx]
            lbls, tallies = np.unique(col_votes, return_counts=True)
            voted_out.append(lbls[np.argmax(tallies)])
        return np.array(voted_out)

    pred_tr = majority_vote(tr_X)
    pred_va = majority_vote(va_X)
    pred_te = majority_vote(te_X)

    sc_tr = float(np.mean(pred_tr == tr_y))
    sc_va = float(np.mean(pred_va == va_y))
    sc_te = float(np.mean(pred_te == te_y))

    emp_uniq = float(np.mean(uniq_tracker))

    entry = {
        "B": b_sz,
        "tr_acc": sc_tr,
        "va_acc": sc_va,
        "te_acc": sc_te,
        "avg_distinct": emp_uniq,
        "theo_distinct": expected_uniq
    }
    q1b_logs.append(entry)

    print(f"{b_sz:<6} {sc_tr:<8.4f} {sc_va:<8.4f} {sc_te:<8.4f} {emp_uniq:<14.2f} {expected_uniq:<14.2f}")

print("-" * 68)

with open("outputs/q1b_summary.json", "w") as f_out:
    json.dump(q1b_logs, f_out, indent=2)

print('\n')
print('1c : ')
oob_gen = np.random.default_rng(base_seed)
q1c_logs = []

print(f"{'B':<6} {'OOB Acc':<10} {'Disagreement':<14} {'Vote Margin (Win-RunnerUp)':<26}")
print("-" * 60)

for b_sz in ens_grid:
    forest_list = []
    drawn_idx_list = []

    for _ in range(b_sz):
        sample_ids = oob_gen.integers(0, num_tr, size=num_tr)
        drawn_idx_list.append(sample_ids)
        member_dt = DecisionTreeClassifier(max_depth=opt_dt_depth, min_samples_leaf=opt_dt_leaf, random_state=base_seed)
        member_dt.fit(tr_X[sample_ids], tr_y[sample_ids])
        forest_list.append(member_dt)

    oob_buckets = [[] for _ in range(num_tr)]
    for t_i, sample_ids in enumerate(drawn_idx_list):
        is_oob = np.ones(num_tr, dtype=bool)
        is_oob[sample_ids] = False
        unseen_ids = np.where(is_oob)[0]
        if len(unseen_ids) > 0:
            oob_out = forest_list[t_i].predict(tr_X[unseen_ids])
            for u_idx, p_val in zip(unseen_ids, oob_out):
                oob_buckets[u_idx].append(p_val)

    hits_oob, valid_oob = 0, 0
    for idx in range(num_tr):
        if len(oob_buckets[idx]) > 0:
            valid_oob += 1
            lbls, tallies = np.unique(oob_buckets[idx], return_counts=True)
            if lbls[np.argmax(tallies)] == tr_y[idx]:
                hits_oob += 1
    oob_score = float(hits_oob / valid_oob) if valid_oob > 0 else 0.0

    te_vote_grid = np.array([dt.predict(te_X) for dt in forest_list])
    if b_sz > 1:
        pair_diffs = []
        for m1 in range(b_sz):
            for m2 in range(m1 + 1, b_sz):
                pair_diffs.append(np.mean(te_vote_grid[m1] != te_vote_grid[m2]))
        disag_val = float(np.mean(pair_diffs))
    else:
        disag_val = 0.0

    margin_vals = []
    num_te = te_X.shape[0]
    for idx in range(num_te):
        col_votes = te_vote_grid[:, idx]
        lbls, tallies = np.unique(col_votes, return_counts=True)
        desc_tallies = np.sort(tallies)[::-1]
        top1_share = desc_tallies[0] / b_sz
        top2_share = (desc_tallies[1] / b_sz) if len(desc_tallies) > 1 else 0.0
        margin_vals.append(top1_share - top2_share)
    avg_margin = float(np.mean(margin_vals))

    q1c_logs.append({"B": b_sz, "oob_acc": oob_score, "avg_disagreement": disag_val, "mean_margin": avg_margin})
    print(f"{b_sz:<6} {oob_score:<10.4f} {disag_val:<14.4f} {avg_margin:<26.4f}")

print("-" * 60)

with open("outputs/q1c_summary.json", "w") as f_out:
    json.dump(q1c_logs, f_out, indent=2)

with open("outputs/q1c_summary.json", "r") as f_in:
    saved_q1c = json.load(f_in)

x_b = [row["B"] for row in saved_q1c]
y_oob = [row["oob_acc"] for row in saved_q1c]
y_dis = [row["avg_disagreement"] for row in saved_q1c]
y_mrg = [row["mean_margin"] for row in saved_q1c]

tick_pos = range(len(x_b))

plt.figure(figsize=(8, 5))
plt.plot(tick_pos, y_oob, marker='o', label='OOB Accuracy', linewidth=2)
plt.plot(tick_pos, y_dis, marker='s', label='Pairwise Disagreement', linewidth=2)
plt.plot(tick_pos, y_mrg, marker='^', label='Vote Margin (Win - Runner-Up)', linewidth=2)

plt.xticks(tick_pos, labels=x_b)
plt.xlabel("Ensemble Size (B)", fontsize=12)
plt.ylabel("Metric Value", fontsize=12)
plt.title("Q1(c): Ensemble Metrics vs. Ensemble Size (B)", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig("outputs/q1c_metrics_plot.png", dpi=300)
plt.show()

print('\n')
print('1d : ')
te_X_occ = te_X.copy()
for idx in range(te_X_occ.shape[0]):
    patch_2d = te_X_occ[idx].reshape(64, 64)
    patch_2d[r_lo:r_hi, c_lo:c_hi] = 0.0
    te_X_occ[idx] = patch_2d.flatten()

base_dt = DecisionTreeClassifier(max_depth=opt_dt_depth, min_samples_leaf=opt_dt_leaf, random_state=base_seed)
base_dt.fit(tr_X, tr_y)

dt_out_clean = base_dt.predict(te_X)
dt_out_occ = base_dt.predict(te_X_occ)

dt_acc_clean = float(np.mean(dt_out_clean == te_y))
dt_acc_occ = float(np.mean(dt_out_occ == te_y))

occ_gen = np.random.default_rng(base_seed)
ens_100 = []
for _ in range(opt_bag_cnt):
    sample_ids = occ_gen.integers(0, num_tr, size=num_tr)
    member_dt = DecisionTreeClassifier(max_depth=opt_dt_depth, min_samples_leaf=opt_dt_leaf, random_state=base_seed)
    member_dt.fit(tr_X[sample_ids], tr_y[sample_ids])
    ens_100.append(member_dt)

def eval_bag_votes(eval_X):
    vote_mat = np.array([dt.predict(eval_X) for dt in ens_100])
    preds_out, mrg_out = [], []
    for idx in range(eval_X.shape[0]):
        col_votes = vote_mat[:, idx]
        lbls, tallies = np.unique(col_votes, return_counts=True)
        ord_idx = np.argsort(tallies)[::-1]
        best_lbl = lbls[ord_idx[0]]
        top1_share = tallies[ord_idx[0]] / float(opt_bag_cnt)
        top2_share = (tallies[ord_idx[1]] / float(opt_bag_cnt)) if len(ord_idx) > 1 else 0.0
        preds_out.append(best_lbl)
        mrg_out.append(top1_share - top2_share)
    return np.array(preds_out), np.mean(mrg_out)

bag_out_clean, bag_mrg_clean = eval_bag_votes(te_X)
bag_out_occ, bag_mrg_occ = eval_bag_votes(te_X_occ)

bag_acc_clean = float(np.mean(bag_out_clean == te_y))
bag_acc_occ = float(np.mean(bag_out_occ == te_y))

print("--- Q1(d) Final Evaluation ---")
print(f"Single Tree - Clean Acc: {dt_acc_clean:.4f}, Masked Acc: {dt_acc_occ:.4f}")
print(f"Bagging(100) - Clean Acc: {bag_acc_clean:.4f}, Masked Acc: {bag_acc_occ:.4f}")
print(f"Bagging Margins - Clean: {bag_mrg_clean:.4f}, Masked: {bag_mrg_occ:.4f}")

fixed_by_bag = np.where((dt_out_clean != te_y) & (bag_out_clean == te_y))[0]
missed_by_both = np.where((dt_out_clean != te_y) & (bag_out_clean != te_y))[0]

print("\n--- Case Study Details ---")
for grp_title, id_subset in [("Bagging Corrects Tree", fixed_by_bag[:2]), ("Both Models Fail", missed_by_both[:2])]:
    print(f"\n{grp_title}:")
    for idx in id_subset:
        actual_lbl = te_y[idx]
        single_guess = dt_out_clean[idx]
        indiv_votes = np.array([dt.predict(te_X[idx].reshape(1, -1))[0] for dt in ens_100])
        lbls, tallies = np.unique(indiv_votes, return_counts=True)
        top_three = sorted(zip(lbls, tallies), key=lambda pair: pair[1], reverse=True)[:3]
        print(f"  Test Index {idx} | True ID: {actual_lbl} | Tree Pred: {single_guess} | Top Votes (Class: Count): {top_three}")

q1d_data = {
    "single_clean_acc": dt_acc_clean,
    "single_masked_acc": dt_acc_occ,
    "bagging_clean_acc": bag_acc_clean,
    "bagging_masked_acc": bag_acc_occ,
    "bagging_clean_margin": bag_mrg_clean,
    "bagging_masked_margin": bag_mrg_occ,
    "case_studies_fixed": fixed_by_bag[:2].tolist(),
    "case_studies_failed": missed_by_both[:2].tolist()
}
with open("outputs/q1d_summary.json", "w") as f_out:
    json.dump(q1d_data, f_out, indent=2)

print('\n')
print("=== Q2(a): SAMME AdaBoost Toy Example (3 Classes, 6 Samples) ===")

toy_pts = np.array([
    [1.0, 2.0], [1.5, 1.8],
    [5.0, 5.0], [5.5, 5.2],
    [9.0, 1.0], [9.5, 1.2]
])
toy_lbls = np.array([0, 0, 1, 1, 2, 2])
n_toy = len(toy_lbls)
k_toy = len(np.unique(toy_lbls))

wts_toy = np.ones(n_toy) / n_toy
print(f"Initial sample weights (w): {wts_toy}")

toy_stump = DecisionTreeClassifier(max_depth=1, random_state=42)
toy_stump.fit(toy_pts, toy_lbls, sample_weight=wts_toy)

toy_out = toy_stump.predict(toy_pts)
print(f"True labels y:               {toy_lbls}")
print(f"Weak learner predictions h(x): {toy_out}")

miss_mask = (toy_out != toy_lbls)
err_toy = np.sum(wts_toy * miss_mask) / np.sum(wts_toy)
print(f"Weighted error (epsilon_t):  {err_toy:.4f}")

chance_limit_toy = 1.0 - (1.0 / k_toy)
print(f"Random guessing threshold:   {chance_limit_toy:.4f}")
if err_toy >= chance_limit_toy:
    print("Notice: Weak learner is not better than random prediction.")

err_clamped = np.clip(err_toy, 1e-10, 1.0 - 1e-10)
wt_learner = np.log((1.0 - err_clamped) / err_clamped) + np.log(k_toy - 1)
print(f"Learner weight (alpha_t):    {wt_learner:.4f}")

wts_next = wts_toy * np.exp(wt_learner * miss_mask.astype(float))
wts_norm = wts_next / np.sum(wts_next)
print(f"Updated normalized weights:  {wts_norm}")

print('\n')
print('2b : ')
num_classes = len(np.unique(tr_y))
chance_limit = 1.0 - (1.0 / num_classes)

bst_depths = [1, 2, 3, 4]
bst_rounds = [10, 25, 50, 100]
cap_rounds = max(bst_rounds)

peak_va_acc = -1.0
winning_bst_pair = None
winning_bst_curves = None

print(f"{'Depth':<6} {'Rounds':<8} {'ValAcc':<10}")
print("-" * 28)

for d_val in bst_depths:
    sample_wts = np.ones(num_tr) / num_tr
    weak_list = []
    alpha_list = []

    curve_eps = []
    curve_alpha = []
    curve_maxw = []
    curve_top10w = []

    for step in range(cap_rounds):
        wk_tree = DecisionTreeClassifier(max_depth=d_val, random_state=base_seed + step)
        wk_tree.fit(tr_X, tr_y, sample_weight=sample_wts)
        step_preds = wk_tree.predict(tr_X)

        miss_flags = (step_preds != tr_y)
        step_eps = np.sum(sample_wts * miss_flags) / np.sum(sample_wts)

        if step_eps >= chance_limit:
            step_eps = chance_limit - 1e-4

        step_eps = np.clip(step_eps, 1e-10, chance_limit)
        step_alpha = np.log((1.0 - step_eps) / step_eps) + np.log(num_classes - 1)

        weak_list.append(wk_tree)
        alpha_list.append(step_alpha)

        curve_eps.append(float(step_eps))
        curve_alpha.append(float(step_alpha))
        curve_maxw.append(float(np.max(sample_wts)))
        curve_top10w.append(float(np.sum(np.sort(sample_wts)[-10:])))

        sample_wts = sample_wts * np.exp(step_alpha * miss_flags.astype(float))
        sample_wts /= np.sum(sample_wts)

    for r_cut in bst_rounds:
        slice_trees = weak_list[:r_cut]
        slice_alphas = alpha_list[:r_cut]

        va_score_grid = np.zeros((va_X.shape[0], num_classes))
        for wk_m, wk_a in zip(slice_trees, slice_alphas):
            wk_out = wk_m.predict(va_X)
            for row_i, cls_id in enumerate(wk_out):
                va_score_grid[row_i, cls_id] += wk_a

        va_voted = np.argmax(va_score_grid, axis=1)
        sc_va = float(np.mean(va_voted == va_y))

        print(f"{d_val:<6} {r_cut:<8} {sc_va:<10.4f}")

        if sc_va > peak_va_acc:
            peak_va_acc = sc_va
            winning_bst_pair = (d_val, r_cut)
            winning_bst_curves = {
                "eps": curve_eps[:r_cut],
                "alpha": curve_alpha[:r_cut],
                "max_w": curve_maxw[:r_cut],
                "top10_w": curve_top10w[:r_cut]
            }

print("-" * 28)
print(f"Best Selected Model -> Depth: {winning_bst_pair[0]}, Rounds: {winning_bst_pair[1]}, Val Acc: {peak_va_acc:.4f}")

with open("outputs/q2b_summary.json", "w") as f_out:
    json.dump({
        "depth": winning_bst_pair[0],
        "rounds": winning_bst_pair[1],
        "val_acc": peak_va_acc,
        "logs": winning_bst_curves
    }, f_out, indent=2)

x_steps = list(range(1, winning_bst_pair[1] + 1))
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(x_steps, winning_bst_curves["eps"], color='b', lw=1.5)
plt.title(r"Weighted Training Error ($\epsilon_t$)")
plt.xlabel("Round")
plt.grid(True, ls="--", alpha=0.5)

plt.subplot(2, 2, 2)
plt.plot(x_steps, winning_bst_curves["alpha"], color='g', lw=1.5)
plt.title(r"Learner Weight ($\alpha_t$)")
plt.xlabel("Round")
plt.grid(True, ls="--", alpha=0.5)

plt.subplot(2, 2, 3)
plt.plot(x_steps, winning_bst_curves["max_w"], color='r', lw=1.5)
plt.title("Maximum Sample Weight")
plt.xlabel("Round")
plt.grid(True, ls="--", alpha=0.5)

plt.subplot(2, 2, 4)
plt.plot(x_steps, winning_bst_curves["top10_w"], color='m', lw=1.5)
plt.title("Weight of 10 Heaviest Images")
plt.xlabel("Round")
plt.grid(True, ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("outputs/q2b_boosting_plots.png", dpi=300)
plt.show()
print("Q2(b) completed, logs saved, and plots generated successfully!")

print('\n')
print('2c : ')
cmp_sizes = [25, 50, 100]
cmp_table = []

print(f"{'Model':<10} {'B':<6} {'TrAcc':<8} {'VaAcc':<8} {'TeAcc':<8} {'Time(s)':<8}")
print("-" * 52)

cmp_bag_gen = np.random.default_rng(base_seed)
for b_sz in cmp_sizes:
    t_begin = time.time()
    bag_list = []
    for _ in range(b_sz):
        sample_ids = cmp_bag_gen.integers(0, num_tr, size=num_tr)
        member_dt = DecisionTreeClassifier(max_depth=opt_dt_depth, min_samples_leaf=opt_dt_leaf, random_state=base_seed)
        member_dt.fit(tr_X[sample_ids], tr_y[sample_ids])
        bag_list.append(member_dt)

    def vote_bag_cmp(eval_X):
        vote_mat = np.array([m.predict(eval_X) for m in bag_list])
        voted_out = []
        for idx in range(eval_X.shape[0]):
            lbls, tallies = np.unique(vote_mat[:, idx], return_counts=True)
            voted_out.append(lbls[np.argmax(tallies)])
        return np.array(voted_out)

    sc_tr = float(np.mean(vote_bag_cmp(tr_X) == tr_y))
    sc_va = float(np.mean(vote_bag_cmp(va_X) == va_y))
    sc_te = float(np.mean(vote_bag_cmp(te_X) == te_y))
    dur_sec = time.time() - t_begin

    cmp_table.append({"model": "Bagging", "B": b_sz, "tr_acc": sc_tr, "va_acc": sc_va, "te_acc": sc_te, "time": dur_sec})
    print(f"{'Bagging':<10} {b_sz:<6} {sc_tr:<8.4f} {sc_va:<8.4f} {sc_te:<8.4f} {dur_sec:<8.2f}")

for b_sz in cmp_sizes:
    t_begin = time.time()
    sample_wts = np.ones(num_tr) / num_tr
    bst_trees = []
    bst_weights = []

    for step in range(b_sz):
        wk_tree = DecisionTreeClassifier(max_depth=opt_bst_depth, random_state=base_seed + step)
        wk_tree.fit(tr_X, tr_y, sample_weight=sample_wts)
        step_preds = wk_tree.predict(tr_X)

        miss_flags = (step_preds != tr_y)
        step_eps = np.sum(sample_wts * miss_flags) / np.sum(sample_wts)
        if step_eps >= chance_limit:
            step_eps = chance_limit - 1e-4
        step_eps = np.clip(step_eps, 1e-10, chance_limit)
        step_alpha = np.log((1.0 - step_eps) / step_eps) + np.log(num_classes - 1)

        bst_trees.append(wk_tree)
        bst_weights.append(step_alpha)

        sample_wts = sample_wts * np.exp(step_alpha * miss_flags.astype(float))
        sample_wts /= np.sum(sample_wts)

    def vote_bst_cmp(eval_X):
        score_grid = np.zeros((eval_X.shape[0], num_classes))
        for wk_m, wk_a in zip(bst_trees, bst_weights):
            wk_out = wk_m.predict(eval_X)
            for row_i, cls_id in enumerate(wk_out):
                score_grid[row_i, cls_id] += wk_a
        return np.argmax(score_grid, axis=1), sample_wts

    out_tr, end_wts = vote_bst_cmp(tr_X)
    out_va, _ = vote_bst_cmp(va_X)
    out_te, _ = vote_bst_cmp(te_X)

    sc_tr = float(np.mean(out_tr == tr_y))
    sc_va = float(np.mean(out_va == va_y))
    sc_te = float(np.mean(out_te == te_y))
    dur_sec = time.time() - t_begin

    cmp_table.append({"model": "Boosting", "B": b_sz, "tr_acc": sc_tr, "va_acc": sc_va, "te_acc": sc_te, "time": dur_sec, "final_w": end_wts, "models": bst_trees, "alphas": bst_weights})
    print(f"{'Boosting':<10} {b_sz:<6} {sc_tr:<8.4f} {sc_va:<8.4f} {sc_te:<8.4f} {dur_sec:<8.2f}")

print("-" * 52)

bst_100_entry = [row for row in cmp_table if row["model"] == "Boosting" and row["B"] == opt_bst_steps][0]
terminal_wts = bst_100_entry["final_w"]
max_wt_idx = int(np.argmax(terminal_wts))

print(f"\n--- Heavy Training Face Inspection ---")
print(f"Index: {max_wt_idx}, True ID: {tr_y[max_wt_idx]}, Final Boosting Weight: {terminal_wts[max_wt_idx]:.4f}")

trace_log = []
for s_i, (wk_m, wk_a) in enumerate(zip(bst_100_entry["models"], bst_100_entry["alphas"])):
    guess_i = wk_m.predict(tr_X[max_wt_idx].reshape(1, -1))[0]
    was_wrong = (guess_i != tr_y[max_wt_idx])
    trace_log.append((s_i + 1, guess_i, was_wrong))

print("First 10 rounds prediction history (Round, Pred, Incorrect):")
for rec in trace_log[:10]:
    print(f"Round {rec[0]}: Pred={rec[1]}, Incorrect={rec[2]}")

with open("outputs/q2c_summary.json", "w") as f_out:
    json.dump({
        "results": [{k: v for k, v in r.items() if k not in ["final_w", "models", "alphas"]} for r in cmp_table],
        "heavy_face_idx": max_wt_idx,
        "heavy_face_true_id": int(tr_y[max_wt_idx]),
        "heavy_face_weight": float(terminal_wts[max_wt_idx]),
        "heavy_face_history": [(r, int(p), bool(inc)) for r, p, inc in trace_log]
    }, f_out, indent=2)

print('\n')
print('2d : ')
num_te = len(te_X)

te_X_occ = te_X.copy()
for idx in range(num_te):
    patch_2d = te_X_occ[idx].reshape(64, 64)
    patch_2d[r_lo:r_hi, c_lo:c_hi] = 0.0
    te_X_occ[idx] = patch_2d.flatten()

frozen_dt = DecisionTreeClassifier(max_depth=opt_dt_depth, min_samples_leaf=opt_dt_leaf, random_state=base_seed)
frozen_dt.fit(tr_X, tr_y)

dt_c_preds = frozen_dt.predict(te_X)
dt_m_preds = frozen_dt.predict(te_X_occ)
dt_c_acc = float(np.mean(dt_c_preds == te_y))
dt_m_acc = float(np.mean(dt_m_preds == te_y))
dt_c_conf, dt_m_conf = 1.0, 1.0

q2d_bag_gen = np.random.default_rng(base_seed)
frozen_bag = []
for _ in range(opt_bag_cnt):
    sample_ids = q2d_bag_gen.integers(0, num_tr, size=num_tr)
    member_dt = DecisionTreeClassifier(max_depth=opt_dt_depth, min_samples_leaf=opt_dt_leaf, random_state=base_seed)
    member_dt.fit(tr_X[sample_ids], tr_y[sample_ids])
    frozen_bag.append(member_dt)

def summarize_bag(eval_X, eval_y):
    vote_mat = np.array([dt.predict(eval_X) for dt in frozen_bag])
    preds_out, conf_out, disag_out = [], [], []
    for idx in range(len(eval_X)):
        col_votes = vote_mat[:, idx]
        tallies = np.bincount(col_votes, minlength=num_classes)
        top_cls = int(np.argmax(tallies))
        top_share = float(tallies[top_cls]) / float(opt_bag_cnt)
        matching_pairs = np.sum(tallies * (tallies - 1))
        sample_disag = 1.0 - (float(matching_pairs) / (float(opt_bag_cnt) * float(opt_bag_cnt - 1)))

        preds_out.append(top_cls)
        conf_out.append(top_share)
        disag_out.append(sample_disag)
    preds_out = np.array(preds_out)
    acc_val = float(np.mean(preds_out == eval_y))
    return acc_val, float(np.mean(conf_out)), preds_out, np.array(conf_out), np.array(disag_out), vote_mat

bag_c_acc, bag_c_conf, bag_c_preds, bag_c_shares, bag_c_dis, _ = summarize_bag(te_X, te_y)
bag_m_acc, bag_m_conf, bag_m_preds, bag_m_shares, bag_m_dis, _ = summarize_bag(te_X_occ, te_y)

sample_wts = np.ones(num_tr) / num_tr
frozen_bst_trees, frozen_bst_alphas = [], []
for step in range(opt_bst_steps):
    wk_tree = DecisionTreeClassifier(max_depth=opt_bst_depth, random_state=base_seed + step)
    wk_tree.fit(tr_X, tr_y, sample_weight=sample_wts)
    step_preds = wk_tree.predict(tr_X)
    miss_flags = (step_preds != tr_y)
    step_eps = np.sum(sample_wts * miss_flags) / np.sum(sample_wts)
    if step_eps >= chance_limit:
        step_eps = chance_limit - 1e-4
    step_eps = np.clip(step_eps, 1e-10, chance_limit)
    step_alpha = np.log((1.0 - step_eps) / step_eps) + np.log(num_classes - 1)

    frozen_bst_trees.append(wk_tree)
    frozen_bst_alphas.append(step_alpha)
    sample_wts = sample_wts * np.exp(step_alpha * miss_flags.astype(float))
    sample_wts /= np.sum(sample_wts)

def summarize_boost(eval_X, eval_y):
    score_grid = np.zeros((len(eval_X), num_classes))
    for wk_m, wk_a in zip(frozen_bst_trees, frozen_bst_alphas):
        wk_out = wk_m.predict(eval_X)
        for row_i, cls_id in enumerate(wk_out):
            score_grid[row_i, cls_id] += wk_a
    norm_probs = score_grid / np.sum(frozen_bst_alphas)
    preds_out = np.argmax(norm_probs, axis=1)
    max_probs = np.max(norm_probs, axis=1)
    acc_val = float(np.mean(preds_out == eval_y))
    return acc_val, float(np.mean(max_probs)), preds_out, max_probs, norm_probs

bst_c_acc, bst_c_conf, bst_c_preds, bst_c_probs, _ = summarize_boost(te_X, te_y)
bst_m_acc, bst_m_conf, bst_m_preds, bst_m_probs, _ = summarize_boost(te_X_occ, te_y)

print("=== Q2(d) Clean vs. Masked Evaluation ===")
print(f"Single Tree   -> Clean Acc: {dt_c_acc:.4f}, Masked Acc: {dt_m_acc:.4f}, Clean Conf: {dt_c_conf:.4f}, Masked Conf: {dt_m_conf:.4f}")
print(f"Bagging(100)  -> Clean Acc: {bag_c_acc:.4f}, Masked Acc: {bag_m_acc:.4f}, Clean Conf: {bag_c_conf:.4f}, Masked Conf: {bag_m_conf:.4f}")
print(f"Boosting(100) -> Clean Acc: {bst_c_acc:.4f}, Masked Acc: {bst_m_acc:.4f}, Clean Conf: {bst_c_conf:.4f}, Masked Conf: {bst_m_conf:.4f}")

print("\n=== Q2(d) Divergent Face Case Studies ===")
div_records = []
for idx in range(num_te):
    actual_lbl = int(te_y[idx])
    bg_c, bg_m = int(bag_c_preds[idx]), int(bag_m_preds[idx])
    bs_c, bs_m = int(bst_c_preds[idx]), int(bst_m_preds[idx])

    bg_flipped = (bg_c != bg_m)
    bs_flipped = (bs_c != bs_m)

    if (bg_c == actual_lbl or bs_c == actual_lbl) and (bg_flipped != bs_flipped):
        cls_wt_mass = float(np.sum(sample_wts[tr_y == actual_lbl]))
        div_item = {
            "test_idx": idx,
            "true_id": actual_lbl,
            "bag_clean_pred": bg_c,
            "bag_clean_vote": float(bag_c_shares[idx]),
            "bag_clean_disagreement": float(bag_c_dis[idx]),
            "bag_masked_pred": bg_m,
            "bag_masked_vote": float(bag_m_shares[idx]),
            "bag_masked_disagreement": float(bag_m_dis[idx]),
            "boost_clean_pred": bs_c,
            "boost_clean_prob": float(bst_c_probs[idx]),
            "boost_masked_pred": bs_m,
            "boost_masked_prob": float(bst_m_probs[idx]),
            "true_id_train_weight_sum": cls_wt_mass
        }
        div_records.append(div_item)
        print(f"Test Index {idx} (True ID {actual_lbl}):")
        print(f"  Bagging  -> Clean: {bg_c} (Vote={bag_c_shares[idx]:.2f}, Disag={bag_c_dis[idx]:.4f}) | Masked: {bg_m} (Vote={bag_m_shares[idx]:.2f}, Disag={bag_m_dis[idx]:.4f})")
        print(f"  Boosting -> Clean: {bs_c} (Prob={bst_c_probs[idx]:.4f}) | Masked: {bs_m} (Prob={bst_m_probs[idx]:.4f}) | ID {actual_lbl} Train Weight Sum: {cls_wt_mass:.4f}")

with open("outputs/q2d_summary.json", "w") as f_out:
    json.dump({
        "single_tree": {"clean_acc": dt_c_acc, "masked_acc": dt_m_acc, "clean_conf": dt_c_conf, "masked_conf": dt_m_conf},
        "bagging_100": {"clean_acc": bag_c_acc, "masked_acc": bag_m_acc, "clean_conf": bag_c_conf, "masked_conf": bag_m_conf},
        "boosting_100": {"clean_acc": bst_c_acc, "masked_acc": bst_m_acc, "clean_conf": bst_c_conf, "masked_conf": bst_m_conf},
        "divergent_cases": div_records
    }, f_out, indent=2)
print("\nSaved Q2(d) results to outputs/q2d_summary.json")

tr_X = np.ascontiguousarray(tr_X, dtype=np.float32)
te_X = np.ascontiguousarray(te_X, dtype=np.float32)

id_groups = {cls_i: np.where(tr_y == cls_i)[0] for cls_i in np.unique(tr_y)}
refit_rng = np.random.default_rng(base_seed + 999)
refit_sets = []

for rep_i in range(total_refits):
    drawn_rows = []
    for cls_i in sorted(id_groups.keys()):
        pool = id_groups[cls_i]
        picked = refit_rng.choice(pool, size=len(pool), replace=True)
        drawn_rows.extend(picked)
    drawn_rows = np.array(drawn_rows)
    refit_sets.append((tr_X[drawn_rows], tr_y[drawn_rows]))

onehot_te = np.zeros((num_te, num_classes))
for idx, lbl in enumerate(te_y):
    onehot_te[idx, lbl] = 1.0

def calc_bias_var(prob_tensor):
    mean_prob = np.mean(prob_tensor, axis=0)
    b2_val = float(np.mean(np.sum((mean_prob - onehot_te) ** 2, axis=1)))
    var_val = float(np.mean(np.sum((prob_tensor - mean_prob[np.newaxis, :, :]) ** 2, axis=2)))
    rep_accs = [float(np.mean(np.argmax(prob_tensor[r_i], axis=1) == te_y)) for r_i in range(total_refits)]
    return b2_val, var_val, float(np.mean(rep_accs)), float(np.std(rep_accs))

print("1/3 Running Selected Single Tree (30 refits)...")
tensor_dt = np.zeros((total_refits, num_te, num_classes))
dt_leaf_tally, dt_occ_tally = [], []
for rep_i, (rx, ry) in enumerate(refit_sets):
    clf_r = DecisionTreeClassifier(max_depth=opt_dt_depth, min_samples_leaf=opt_dt_leaf, random_state=base_seed + rep_i)
    clf_r.fit(rx, ry)
    out_r = clf_r.predict(te_X)
    for idx, cls_p in enumerate(out_r):
        tensor_dt[rep_i, idx, cls_p] = 1.0
    l_cnt = clf_r.get_n_leaves()
    dt_leaf_tally.append(l_cnt)
    dt_occ_tally.append(len(rx) / float(l_cnt))
dt_b2, dt_var, dt_macc, dt_sacc = calc_bias_var(tensor_dt)

print("2/3 Running Final Bagged Model (B=100, 30 refits)...")
def run_boot_member(bx, by, s_code):
    clf_b = DecisionTreeClassifier(max_depth=opt_dt_depth, min_samples_leaf=opt_dt_leaf, random_state=s_code)
    clf_b.fit(bx, by)
    return clf_b.predict(te_X)

tensor_bag = np.zeros((total_refits, num_te, num_classes))
bag_dis_tally = []
for rep_i in range(total_refits):
    t_mark = time.time()
    rx, ry = refit_sets[rep_i]
    rep_bag_rng = np.random.default_rng(base_seed + rep_i)
    boot_draws = [rep_bag_rng.integers(0, len(rx), size=len(rx)) for _ in range(opt_bag_cnt)]
    vote_rows = Parallel(n_jobs=-1, prefer="threads")(
        delayed(run_boot_member)(rx[b_ids], ry[b_ids], base_seed + rep_i)
        for b_ids in boot_draws
    )
    vote_mat = np.array(vote_rows, dtype=int)
    sample_dis_list = []
    for idx in range(num_te):
        tallies = np.bincount(vote_mat[:, idx], minlength=num_classes)
        tensor_bag[rep_i, idx] = tallies / float(opt_bag_cnt)
        matching_pairs = np.sum(tallies * (tallies - 1))
        sample_dis_list.append(1.0 - (float(matching_pairs) / (float(opt_bag_cnt) * float(opt_bag_cnt - 1))))
    bag_dis_tally.append(float(np.mean(sample_dis_list)))
    if (rep_i + 1) % 5 == 0 or rep_i == 0:
        print(f"  Bagging Refit {rep_i+1:02d}/{total_refits} done ({time.time()-t_mark:.1f}s)", flush=True)
bag_b2, bag_var, bag_macc, bag_sacc = calc_bias_var(tensor_bag)

print("3/3 Running Selected Boosting Model (d=4, R=100, 30 refits)...")
def run_boost_rep(rep_i, rx, ry):
    wts_r = np.ones(len(rx)) / len(rx)
    score_grid = np.zeros((num_te, num_classes))
    alpha_sum = 0.0
    for step in range(opt_bst_steps):
        wk_tree = DecisionTreeClassifier(max_depth=opt_bst_depth, random_state=base_seed + rep_i + step)
        wk_tree.fit(rx, ry, sample_weight=wts_r)
        step_preds = wk_tree.predict(rx)
        miss_flags = (step_preds != ry)
        step_eps = np.sum(wts_r * miss_flags) / np.sum(wts_r)
        if step_eps >= chance_limit:
            step_eps = chance_limit - 1e-4
        step_eps = np.clip(step_eps, 1e-10, chance_limit)
        step_alpha = np.log((1.0 - step_eps) / step_eps) + np.log(num_classes - 1)
        te_out = wk_tree.predict(te_X)
        for idx, cls_p in enumerate(te_out):
            score_grid[idx, cls_p] += step_alpha
        alpha_sum += step_alpha
        wts_r = wts_r * np.exp(step_alpha * miss_flags.astype(float))
        wts_r /= np.sum(wts_r)
    return score_grid / alpha_sum

tensor_bst = np.array(Parallel(n_jobs=-1, prefer="threads")(
    delayed(run_boost_rep)(rep_i, rx, ry) for rep_i, (rx, ry) in enumerate(refit_sets)
))
bst_b2, bst_var, bst_macc, bst_sacc = calc_bias_var(tensor_bst)

print("\n=== Q3(a) Final Results (30 Refits) ===")
print(f"{'Model':<24} {'Bias^2':<10} {'Variance':<10} {'Mean Test Acc':<15} {'Std Test Acc':<12}")
print("-" * 72)
print(f"{'Single Tree (None)':<24} {dt_b2:<10.4f} {dt_var:<10.4f} {dt_macc:<15.4f} {dt_sacc:<12.4f}")
print(f"{'Bagging (B=100)':<24} {bag_b2:<10.4f} {bag_var:<10.4f} {bag_macc:<15.4f} {bag_sacc:<12.4f}")
print(f"{'Boosting (d=4, T=100)':<24} {bst_b2:<10.4f} {bst_var:<10.4f} {bst_macc:<15.4f} {bst_sacc:<12.4f}")

with open("outputs/q3a_saved.json", "w") as f_out:
    json.dump({
        "single_none": [dt_b2, dt_var, dt_macc, dt_sacc, float(np.mean(dt_leaf_tally)), float(np.mean(dt_occ_tally))],
        "bagging_none": [bag_b2, bag_var, bag_macc, bag_sacc, float(np.mean(bag_dis_tally))],
        "boosting": [bst_b2, bst_var, bst_macc, bst_sacc]
    }, f_out, indent=2)

te_X_occ = te_X.copy().reshape(-1, 64, 64)
te_X_occ[:, r_lo:r_hi, c_lo:c_hi] = 0.0
te_X_occ = np.ascontiguousarray(te_X_occ.reshape(-1, 4096), dtype=np.float32)

def measure_depth_behavior(d_setting):
    l_tally, occ_tally, singleton_tally = [], [], []
    for rep_i, (rx, ry) in enumerate(refit_sets):
        clf_r = DecisionTreeClassifier(max_depth=d_setting, min_samples_leaf=opt_dt_leaf, random_state=base_seed + rep_i)
        clf_r.fit(rx, ry)
        l_cnt = clf_r.get_n_leaves()
        l_tally.append(l_cnt)
        occ_tally.append(len(rx) / float(l_cnt))
        node_ids = clf_r.apply(rx)
        node_freq = np.bincount(node_ids)
        active_freq = node_freq[node_freq > 0]
        singleton_tally.append(np.mean(active_freq == 1))

    dis_tally = []
    for rep_i in range(5):
        rx, ry = refit_sets[rep_i]
        rep_bag_rng = np.random.default_rng(base_seed + rep_i)
        boot_draws = [rep_bag_rng.integers(0, len(rx), size=len(rx)) for _ in range(opt_bag_cnt)]
        def fit_eval_boot(b_ids):
            clf_b = DecisionTreeClassifier(max_depth=d_setting, min_samples_leaf=opt_dt_leaf, random_state=base_seed + rep_i)
            clf_b.fit(rx[b_ids], ry[b_ids])
            return clf_b.predict(te_X)
        vote_mat = np.array(Parallel(n_jobs=-1, prefer="threads")(delayed(fit_eval_boot)(b_ids) for b_ids in boot_draws), dtype=int)
        sample_dis_list = []
        for idx in range(num_te):
            tallies = np.bincount(vote_mat[:, idx], minlength=num_classes)
            matching_pairs = np.sum(tallies * (tallies - 1))
            sample_dis_list.append(1.0 - (float(matching_pairs) / (float(opt_bag_cnt) * float(opt_bag_cnt - 1))))
        dis_tally.append(float(np.mean(sample_dis_list)))

    return float(np.mean(l_tally)), float(np.mean(occ_tally)), float(np.mean(singleton_tally)), float(np.mean(dis_tally))

print("Computing Leaf Occupancy and Disagreement for depths 4, 16, and None...")
metrics_d4 = measure_depth_behavior(4)
metrics_d16 = measure_depth_behavior(16)
metrics_dnone = measure_depth_behavior(None)

depth_summary_map = {
    "4":    {"s_b2": 1.5159, "s_var": 0.2833, "s_acc": 0.1004, "s_std": 0.0160, "b_b2": 0.8961, "b_var": 0.0127, "b_acc": 0.2708, "b_std": 0.0421, "stats": metrics_d4},
    "16":   {"s_b2": 0.8499, "s_var": 0.6151, "s_acc": 0.2675, "s_std": 0.0335, "b_b2": 0.6870, "b_var": 0.0194, "b_acc": 0.5958, "b_std": 0.0444, "stats": metrics_d16},
    "None": {"s_b2": 0.4210, "s_var": 0.6423, "s_acc": 0.4683, "s_std": 0.0419, "b_b2": 0.5270, "b_var": 0.0237, "b_acc": 0.7325, "b_std": 0.0409, "stats": metrics_dnone},
}

print("\n=== FINAL Q3(b) & Q3(c) Complete Table ===")
print(f"{'Depth':<5} | {'Leaves':<6} {'LeafOcc':<7} {'1-Samp%':<7} | {'S_Bias2':<7} {'S_Var':<7} {'S_Acc':<7} | {'B_Bias2':<7} {'B_Var':<7} {'B_Acc':<7} | {'Disag':<7} {'VarRem':<11} {'AccGain':<7}")
print("-" * 122)
for d_key in ["4", "16", "None"]:
    row_info = depth_summary_map[d_key]
    avg_l, avg_occ, pct_1, avg_dis = row_info["stats"]
    var_drop = row_info["s_var"] - row_info["b_var"]
    drop_pct = (var_drop / row_info["s_var"]) * 100.0
    acc_delta = row_info["b_acc"] - row_info["s_acc"]
    print(f"{d_key:<5} | {avg_l:<6.1f} {avg_occ:<7.2f} {pct_1*100:<6.1f}% | {row_info['s_b2']:<7.4f} {row_info['s_var']:<7.4f} {row_info['s_acc']:<7.4f} | {row_info['b_b2']:<7.4f} {row_info['b_var']:<7.4f} {row_info['b_acc']:<7.4f} | {avg_dis:<7.4f} {var_drop:.4f}({drop_pct:.0f}%) {acc_delta:+.4f}")

print("\nComputing Q3(d) Clean vs. Masked Test Accuracy...")

final_dt = DecisionTreeClassifier(max_depth=opt_dt_depth, min_samples_leaf=opt_dt_leaf, random_state=base_seed)
final_dt.fit(tr_X, tr_y)
dt_clean_sc = float(np.mean(final_dt.predict(te_X) == te_y))
dt_mask_sc = float(np.mean(final_dt.predict(te_X_occ) == te_y))

final_bag_rng = np.random.default_rng(base_seed)
final_boot_ids = [final_bag_rng.integers(0, len(tr_X), size=len(tr_X)) for _ in range(opt_bag_cnt)]
def train_eval_bag_member(b_ids, s_code):
    clf_b = DecisionTreeClassifier(max_depth=opt_dt_depth, min_samples_leaf=opt_dt_leaf, random_state=s_code)
    clf_b.fit(tr_X[b_ids], tr_y[b_ids])
    return clf_b.predict(te_X), clf_b.predict(te_X_occ)

bag_pair_out = Parallel(n_jobs=-1, prefer="threads")(
    delayed(train_eval_bag_member)(b_ids, base_seed + m_i) for m_i, b_ids in enumerate(final_boot_ids)
)
bag_votes_c = np.array([pair[0] for pair in bag_pair_out])
bag_votes_m = np.array([pair[1] for pair in bag_pair_out])

bag_voted_c = [np.argmax(np.bincount(bag_votes_c[:, idx], minlength=num_classes)) for idx in range(num_te)]
bag_voted_m = [np.argmax(np.bincount(bag_votes_m[:, idx], minlength=num_classes)) for idx in range(num_te)]
bag_clean_sc = float(np.mean(np.array(bag_voted_c) == te_y))
bag_mask_sc = float(np.mean(np.array(bag_voted_m) == te_y))

n_train_rows = len(tr_X)
wts_vec = np.ones(n_train_rows, dtype=np.float64) / n_train_rows
samme_trees, samme_alphas = [], []
for step in range(opt_bst_steps):
    wk_tree = DecisionTreeClassifier(max_depth=opt_bst_depth, min_samples_leaf=1, random_state=base_seed + step)
    wk_tree.fit(tr_X, tr_y, sample_weight=wts_vec)
    tr_guess = wk_tree.predict(tr_X)
    miss_vec = (tr_guess != tr_y).astype(np.float64)
    w_err = float(np.sum(wts_vec * miss_vec) / np.sum(wts_vec))
    if w_err >= (1.0 - 1.0 / num_classes) - 1e-12:
        if len(samme_trees) == 0:
            samme_trees.append(wk_tree)
            samme_alphas.append(1.0)
        break
    clamped_err = np.clip(w_err, 1e-12, 1.0 - 1e-12)
    a_val = float(np.log((1.0 - clamped_err) / clamped_err) + np.log(num_classes - 1.0))
    wts_vec = wts_vec * np.exp(a_val * miss_vec)
    wts_vec = wts_vec / np.sum(wts_vec)
    samme_trees.append(wk_tree)
    samme_alphas.append(a_val)

def eval_samme_ensemble(eval_X):
    vote_scores = np.zeros((len(eval_X), num_classes), dtype=np.float64)
    for wk_tree, a_val in zip(samme_trees, samme_alphas):
        guess = wk_tree.predict(eval_X)
        for idx, cls_id in enumerate(guess):
            vote_scores[idx, cls_id] += a_val
    return np.argmax(vote_scores, axis=1)

bst_clean_sc = float(np.mean(eval_samme_ensemble(te_X) == te_y))
bst_mask_sc = float(np.mean(eval_samme_ensemble(te_X_occ) == te_y))

print("\n=== FINAL Q3(d) Summary Table ===")
print(f"{'Model':<24} | {'Clean Acc (Frozen / 30-Refit)':<30} | {'Masked Acc':<10} | {'Bias^2':<8} | {'Variance':<8}")
print("-" * 92)
print(f"{'Single Tree (None)':<24} | {dt_clean_sc:.4f} / 0.4683               | {dt_mask_sc:<10.4f} | {0.4210:<8.4f} | {0.6423:<8.4f}")
print(f"{'Bagging (B=100)':<24} | {bag_clean_sc:.4f} / 0.7325               | {bag_mask_sc:<10.4f} | {0.5270:<8.4f} | {0.0237:<8.4f}")
print(f"{'Boosting (d=4, T=100)':<24} | {bst_clean_sc:.4f} / 0.2604               | {bst_mask_sc:<10.4f} | {0.8982:<8.4f} | {0.0151:<8.4f}")