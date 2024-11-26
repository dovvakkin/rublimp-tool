import click
import logging
from rublimp.src import scorer as rublimp_scorer
import pandas as pd
import datasets
import pathlib


RUBLIMP_DATASET_NAME = 'RussianNLP/rublimp'
MODEL_TYPES = ['encoder', 'decoder']


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
)


@click.group()
def cli():
    pass


def score_rublimp_subsets(scorer: rublimp_scorer.Scorer, subsets: list[str]) -> pd.DataFrame:
    res = {}
    for subset in subsets:
        dataset = datasets.load_dataset(RUBLIMP_DATASET_NAME, subset)['train'].to_pandas()
        res[subset] = scorer.run(pool=dataset)

    return pd.concat(res, names=['subset']).reset_index(level=0)


def get_subsets(subsets_option: list[str]):
    all_subsets = datasets.get_dataset_config_names(RUBLIMP_DATASET_NAME)
    if len(subsets_option) == 1 and subsets_option[0] == 'all':
        return all_subsets

    not_valid_subset = False
    for subset in subsets_option:
        if subset not in all_subsets:
            logging.error(f'no such subset `{subset}`')
            not_valid_subset = True
    if not_valid_subset:
        logging.error(f'available susbsets are {all_subsets}')
        exit(1)

    return subsets_option


def get_model_name(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.endswith('-ppl-s'):
            return col[:-6]

    raise ValueError('no -ppl-s column in df')


def calculate_accuracies_for_models(dfs: list[pd.DataFrame], model_type: str) -> pd.DataFrame:
    accuracies = {}

    all_subsamples = set()
    for df in dfs:
        all_subsamples.update(df['subset'].unique())

    for df in dfs:
        model = get_model_name(df)
        source_col = f'{model}-ppl-s'
        target_col = f'{model}-ppl-t'

        # https://github.com/RussianNLP/RuBLiMP?tab=readme-ov-file#scoring-with-min-k
        if model_type == 'encoder':
            model_acc = df.groupby('subset').apply(lambda x: (x[source_col] > x[target_col]).mean())
        elif model_type == 'decoder':
            model_acc = df.groupby('subset').apply(lambda x: (x[source_col] < x[target_col]).mean())
        else:
            raise ValueError(f'unknown model type `{model_type}`')

        full_acc = pd.Series(index=all_subsamples, dtype=float)
        full_acc.loc[model_acc.index] = model_acc

        accuracies[model] = full_acc

    result_df = pd.DataFrame(accuracies)
    result_df.index.name = 'subset'

    return result_df.reset_index()


@cli.command()
@click.option(
    '--model-type',
    '-t',
    required=True,
    type=click.Choice(MODEL_TYPES),
    help='model type, used for accuracy calculation: https://github.com/RussianNLP/RuBLiMP?tab=readme-ov-file#scoring-with-min-k',
)
@click.option('--output', '-o', required=True, type=click.Path(path_type=pathlib.Path), help='path to save result csv')
@click.argument('model_scores', nargs=-1)
def compare_results(model_scores: list[str], model_type: str, output: pathlib.Path):
    """Calculate accuracies for models evaled with eval-model

    Arguments:
        MODEL_SCORES: .csv files with scored models
    """
    dfs = [pd.read_csv(model_score) for model_score in model_scores]
    accuracies = calculate_accuracies_for_models(dfs, model_type)
    accuracies.to_csv(output)


@cli.command()
@click.option('--model', '-m', required=True, help='model to eval')
@click.option(
    '--subset',
    '-s',
    required=False,
    help=f'{RUBLIMP_DATASET_NAME} subsets to eval, `all` for all available',
    multiple=True,
)
@click.option('--output', '-o', required=True, type=click.Path(path_type=pathlib.Path), help='path to save result csv')
def eval_model(model: str, subset: list[str], output: pathlib.Path):
    subsets = get_subsets(subset)
    model_scorer = rublimp_scorer.Scorer(model)

    logging.info(f'eval model `{model}` on subsets `{subsets}`')
    model_scores = score_rublimp_subsets(model_scorer, subsets)

    model_scores.to_csv(output)


if __name__ == '__main__':
    cli()
