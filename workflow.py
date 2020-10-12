import spell.client
client = spell.client.from_environment()

train = client.runs.new(
    machine_type="cpu",
    pip_packages=["pandas", "scikit-learn"],
    attached_resources={
        "s3://spell-datasets-share/wta-matches/": "/mnt/wta-matches/"
    },
    command="python train.py"
)
train.wait_status(*client.runs.FINAL)
train.refresh()
if train.status != client.runs.COMPLETE:
    raise OSError(f"Failed at training run {train.id}.")

test = []
for partition in range(2000, 2017):
    r = client.runs.new(
        machine_type="cpu",
        docker_image="residentmario/dask-cpu-workspace:latest",
        attached_resources={
            f"runs/{train.id}/wta-matches-model.joblib": "wta-matches-model.joblib",
            f"s3://spell-datasets-share/wta-matches/wta_matches_{partition}.csv": \
                f"wta_matches_{partition}.csv"
        },
        command=f"python score.py --filename wta_matches_{partition}.csv"
    )
    test.append(r)

for run in test:
    run.wait_status(*client.runs.FINAL)
    run.refresh()
    if run.status != client.runs.COMPLETE:
        raise OSError(f"Failed at scoring run {run.id}.")

print("Finished workflow!")