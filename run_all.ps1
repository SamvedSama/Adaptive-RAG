pip install bert-score
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python qa_generator.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python train_router.py --force
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python build_index.py --force
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python ablation_runner.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python evaluation.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python pareto_curve.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
