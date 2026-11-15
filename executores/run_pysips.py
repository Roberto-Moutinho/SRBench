import argparse, json, time, traceback
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sympy import sympify, count_ops

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/"config/benchmark.json").read_text(encoding="utf-8"))
DATA=ROOT/"datasets/raw"
OUT=ROOT/"results_tuning/PySIPS"
OPS={"C1":["+","-","*","/"],"C2":["+","-","*","/","exp","log","sqrt","pow"]}

def load_pair(name):
    train_path=DATA/f"{name}_train.csv"
    test_path=DATA/f"{name}_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test pair for {name}: {train_path.name}, {test_path.name}")
    tr=pd.read_csv(train_path); te=pd.read_csv(test_path)
    target=CFG["target_column"]
    if target not in tr or target not in te: raise ValueError(f"{name}: target column {target!r} missing")
    features=[c for c in tr.columns if c!=target]
    if features != [c for c in te.columns if c!=target]:
        raise ValueError(f"{name}: train/test feature columns or order differ")
    if not all(pd.api.types.is_numeric_dtype(tr[c]) and pd.api.types.is_numeric_dtype(te[c]) for c in features+[target]):
        raise ValueError(f"{name}: all features and target must be numeric")
    tr=tr.replace([np.inf,-np.inf],np.nan).dropna(subset=features+[target])
    te=te.replace([np.inf,-np.inf],np.nan).dropna(subset=features+[target])
    return tr[features].to_numpy(float),tr[target].to_numpy(float),te[features].to_numpy(float),te[target].to_numpy(float),features

def metrics(y,p):
    return {"r2":float(r2_score(y,p)),"rmse":float(mean_squared_error(y,p)**0.5),
            "mae":float(mean_absolute_error(y,p)),"mse":float(mean_squared_error(y,p))}

def complexity(expr):
    try: return int(count_ops(sympify(str(expr),evaluate=False)))+1
    except Exception: return int(len(str(expr).split()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",action="append")
    ap.add_argument("--class",dest="classes",action="append",choices=["C1","C2"])
    ap.add_argument("--fold",type=int,action="append")
    ap.add_argument("--seed",type=int,action="append")
    ap.add_argument("--max-equation-evals",type=int,default=CFG["equation_evaluation_budget"])
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    datasets=args.dataset or CFG["datasets"]; classes=args.classes or ["C1","C2"]
    folds=args.fold or list(range(1,6)); seeds=args.seed or CFG["seeds"]
    expected=len(datasets)*len(classes)*len(folds)*len(seeds)
    if args.dry_run:
        print(f"Planned runs: {expected}")
        for d in datasets:
            print(d, "READY" if (DATA/f"{d}_train.csv").exists() and (DATA/f"{d}_test.csv").exists() else "MISSING TRAIN/TEST")
        return
    # Validate every pair before launching any expensive fit.
    loaded={d:load_pair(d) for d in datasets}
    try:
        from pysips import PysipsRegressor
    except Exception as e:
        raise SystemExit("PySIPS import failed; install requirements and the local pysips package: "+repr(e))
    OUT.mkdir(parents=True,exist_ok=True)
    for ds,(X,y,Xext,yext,features) in loaded.items():
        splits=list(KFold(n_splits=5,shuffle=True,random_state=CFG["fold_random_state"]).split(X))
        for cls in classes:
            for f in folds:
                if not 1<=f<=5: raise ValueError("fold must be 1..5")
                train_idx,val_idx=splits[f-1]
                Xtr,ytr=X[train_idx],y[train_idx]
                Xval,yval=X[val_idx],y[val_idx]
                for seed in seeds:
                    row={"algorithm":"PySIPS","dataset":ds,"operator_class":cls,"operators":OPS[cls],
                         "fold":f,"seed":seed,"n_train":len(Xtr),"n_validation":len(Xval),
                         "n_external_test":len(Xext),"features":features,"expression":None,
                         "complexity":None,"runtime_seconds":None,"status":"failed","error":None}
                    start=time.perf_counter()
                    try:
                        model=PysipsRegressor(operators=OPS[cls],
                            max_complexity=CFG["pysips"]["max_complexity"],
                            num_particles=CFG["pysips"]["num_particles"],
                            num_mcmc_samples=CFG["pysips"]["num_mcmc_samples"],
                            target_ess=CFG["pysips"]["target_ess"],
                            model_selection=CFG["pysips"]["model_selection"],
                            max_equation_evals=args.max_equation_evals,
                            random_state=int(seed),show_progress_bar=False)
                        model.fit(Xtr,ytr)
                        expr=str(model.get_expression())
                        pred_tr=np.asarray(model.predict(Xtr)).reshape(-1)
                        pred_val=np.asarray(model.predict(Xval)).reshape(-1)
                        pred_ext=np.asarray(model.predict(Xext)).reshape(-1)
                        row.update(expression=expr,complexity=complexity(expr),
                            **{"train_"+k:v for k,v in metrics(ytr,pred_tr).items()},
                            **{"validation_"+k:v for k,v in metrics(yval,pred_val).items()},
                            **{"external_test_"+k:v for k,v in metrics(yext,pred_ext).items()},
                            status="ok")
                    except Exception as e:
                        row["error"]=f"{type(e).__name__}: {e}"
                        traceback.print_exc()
                    row["runtime_seconds"]=time.perf_counter()-start
                    stem=f"{cls}_{ds}_seed_{seed}_fold_{f}"
                    pd.DataFrame([row]).to_csv(OUT/f"{stem}.csv",index=False)
                    (OUT/f"{stem}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
                    print(stem,row["status"],row.get("validation_r2"))
if __name__=="__main__": main()
