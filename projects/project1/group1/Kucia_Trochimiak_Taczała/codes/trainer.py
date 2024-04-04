from dataloader import DataloaderModule
from model import LogisticRegression
from optimizers.base import Base
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

import wandb
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


from stop_property import should_stop_convergence


class Trainer:
    def __init__(
        self,
        model: LogisticRegression,
        dataloader: DataloaderModule,
        optimizer: Base,
        log_wandb: bool = False,
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.log_wandb = log_wandb

    def train(
        self, epochs: int
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        train_dataloader = self.dataloader.train_dataloader()
        losses_step = []
        losses_epoch = []
        accuracy_step = []
        accuracy_epoch = []
        log_likelihood_step = []
        log_likelihood_epoch = []
        step = 0
        previous_epoch_loss = float("inf")

        for epoch in range(epochs):
            losses = []
            accuracies = []
            for x, y in train_dataloader:

                y_hat = self.model.forward(x)
                loss = self.model.loss(y_hat, y).item()
                self.optimizer.backprop(x, y, y_hat)
                accuracy = balanced_accuracy_score(y, (y_hat > 0.5).long())
                # print(accuracy, "accuracy")
                losses.append(loss)
                losses_step.append(loss)
                accuracy_step.append(accuracy)
                accuracies.append(accuracy)

                log_likelihood_step.append(
                    Base.log_likelihood(
                        torch.cat((torch.ones(x.shape[0], 1), x), dim=1),
                        y,
                        beta=torch.cat((self.model.beta0, self.model.beta1), dim=0),
                    )
                )
                if self.log_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "step": step,
                            "train/accuracy_step": accuracy,
                            "train/loss_step": loss,
                        }
                    )

                step += 1

            loss_epoch = sum(losses) / len(losses)
            acc_epoch = sum(accuracies) / len(accuracies)
            last_log_likelihood_epoch = sum(log_likelihood_step) / len(
                log_likelihood_step
            )
            log_likelihood_epoch.append(last_log_likelihood_epoch)
            losses_epoch.append(loss_epoch)
            accuracy_epoch.append(acc_epoch)

            if self.log_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "step": step,
                        "train/accuracy_epoch": acc_epoch,
                        "train/loss_epoch": loss_epoch,
                        "train/log_likelihood_epoch": last_log_likelihood_epoch,
                    }
                )
            change_in_objective = previous_epoch_loss - loss_epoch
            previous_epoch_loss = loss_epoch

            is_stopped = should_stop_convergence(change_in_objective)
            if is_stopped:
                break

        return losses_step, losses_epoch, accuracy_step, accuracy_epoch

    def test(self) -> tuple[float, float]:
        losses = []
        accuracies = []
        test_dataloader = self.dataloader.test_dataloader()

        for x, y in test_dataloader:
            y_hat = self.model.forward(x)
            accuracies.append(balanced_accuracy_score(y, (y_hat > 0.5).long()))
            losses.append(self.model.loss(y_hat, y).item())

        loss = sum(losses) / len(losses)
        accuracy = sum(accuracies) / len(accuracies)
        if self.log_wandb:
            wandb.log({"test/accuracy": accuracy, "test/loss": loss})
        return loss, accuracy

    def evaluate_models(self, optymiser_name):
        dl = self.dataloader.train_dataloader()
        dl_test = self.dataloader.test_dataloader()
        data_list = []
        label_list = []

        data_list_test = []
        label_list_test = []
        for x, y in dl:
            data_list.append(x)
            label_list.append(y)
        all_data = torch.cat(data_list, dim=0)
        all_labels = torch.cat(label_list, dim=0)
        # Convert the PyTorch tensors to numpy arrays
        all_data_C = all_data.numpy()  # This works if all_data is on CPU
        all_labels_C = all_labels.numpy()  # This works if all_labels is on CPU

        for x, y in dl_test:
            data_list_test.append(x)
            label_list_test.append(y)
        X_test = torch.cat(data_list_test, dim=0).numpy()
        y_test = torch.cat(label_list_test, dim=0).numpy()

        lda = LinearDiscriminantAnalysis()
        lda.fit(all_data_C, all_labels_C)

        qda = QuadraticDiscriminantAnalysis()
        qda.fit(all_data_C, all_labels_C)

        tree = DecisionTreeClassifier()
        tree.fit(all_data_C, all_labels_C)

        forest = RandomForestClassifier()
        forest.fit(all_data_C, all_labels_C)
        _, y_pred_lr = self.test()
        y_pred_lda = lda.predict(X_test)
        y_pred_qda = qda.predict(X_test)
        y_pred_tree = tree.predict(X_test)
        y_pred_forest = forest.predict(X_test)
        print(f"Logistic Regression Accuracy:{optymiser_name}", y_pred_lr)
        print("LDA Accuracy:", accuracy_score(y_test, y_pred_lda))
        print("QDA Accuracy:", accuracy_score(y_test, y_pred_qda))
        print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
        print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_forest))
        if self.log_wandb:
            wandb.log(
                {
                    f"test/{optymiser_name}_accuracy": y_pred_lr,
                    "test/LDA_accuracy": accuracy_score(y_test, y_pred_lda),
                    "test/QDA_accuracy": accuracy_score(y_test, y_pred_qda),
                    "test/Decision_Tree_accuracy": accuracy_score(y_test, y_pred_tree),
                    "test/Random_Forest_accuracy": accuracy_score(
                        y_test, y_pred_forest
                    ),
                }
            )
