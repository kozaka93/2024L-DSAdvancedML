from sklearn.pipeline import Pipeline


def get_feature_count(pipeline: Pipeline) -> int:
    return len(pipeline[0][0].transformers_[0][2])


def make_paths(*paths):
    for path in paths:
        path.mkdir(exist_ok=True, parents=True)
