import pandas as pd
import plotly.express as px


# ----------------------------------------- Task 3.2 -----------------------------------------
def boxplot_grouped_balanced_accuracies(df: pd.DataFrame) -> None:
    df_melted = df.melt(id_vars=['dataset', 'method'], var_name='iteration', value_name='balanced_accuracy')

    fig = px.box(df_melted, x='dataset', y='balanced_accuracy', color='method')
    fig.update_layout(
        title_text='Balanced accuracy across methods and datasets',
        xaxis_title='dataset', yaxis_title='balanced accuracy',
        height=600,
        width=1200,
        boxgap=0.5,
        boxgroupgap=0.3
    )
    fig.show()
    fig.write_image("images/balanced_accuracy_across_datasets_and_methods.png")


# ----------------------------------------- Task 3.3 -----------------------------------------
def plot_log_likelihoods(log_likelihoods, dataset_name):
    log_likelihoods = log_likelihoods.reset_index().melt(id_vars='index', var_name='method', value_name='log-likelihood')
    fig = px.line(
        log_likelihoods,
        x='index',
        y='log-likelihood',
        color='method',
        title=f'Log-likelihood per iteration, dataset: {dataset_name}',
        labels={'index': 'iteration', 'log-likelihood': 'log-likelihood'},
        width=500,
        height=500
    )
    fig.update_layout(
        showlegend=True,
        title_y=0.88,
        legend=dict(
            x=0.75,
            y=0.05,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
            ),
            bgcolor="white",
            bordercolor="white"
        ),
        autosize=False,
        margin=dict(
            l=50,
            r=50,
            b=50,
            pad=0
        )
    )
    fig.show()
    fig.write_image(f"images/log_likelihood_{dataset_name}.png")


# ----------------------------------------- Task 3.4 -----------------------------------------
def boxplot_classifiers_balanced_accuracy(df, dataset_name):
    df_melted = df.melt(var_name='method', value_name='value')

    fig = px.box(df_melted, x='method', y='value')
    fig.update_layout(
        title_text=f'Balanced accuracy across classifiers<br>on dataset {dataset_name}',
        xaxis_title='classifier',
        yaxis_title='balanced accuracy',
        autosize=False,
        width=500,
        height=500,
        showlegend=False
    )
    fig.update_yaxes(range=[0.46, 1.01])
    fig.show()
    fig.write_image(f"images/classification_methods_{dataset_name}.png")


# ----------------------------------------- Task 3.5 -----------------------------------------
def boxplot_interaction_balanced_accuracy(df, dataset_name):
    df_melted = df.melt(var_name='method', value_name='value')

    fig = px.box(df_melted, x='value', y='method')
    fig.update_layout(
        title_text=f'Balanced accuracy across methods with and without interactions on dataset {dataset_name}',
        xaxis_title='balanced accuracy',
        yaxis_title='method',
        autosize=False,
        width=1000,
        height=450,
        showlegend=False
    )
    fig.update_xaxes(range=[0.47, 1.01])
    fig.show()
    fig.write_image(f"images/interactions_{dataset_name}.png")
