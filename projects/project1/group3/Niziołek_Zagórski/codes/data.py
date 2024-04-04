from ucimlrepo import fetch_ucirepo
from sklearn.impute import KNNImputer
from collinearity import SelectNonCollinear
import openml

small_database_1_id = 850
small_database_2_id = 267
small_database_3_id = 161
large_database_1_id = 52
large_database_2_id = 547
large_database_3_id = 17
large_database_4_id = 1504
large_database_5_id = 1494
large_database_6_id = 602

collinearity_threshold = 1  # what threshold?


def fetch_data():
    data = {}
    imputer = KNNImputer()
    selector = SelectNonCollinear(collinearity_threshold)

    small1 = fetch_ucirepo(id=small_database_1_id)
    small1["data"]["targets"].iloc[:, 0] = (
        small1["data"]["targets"].iloc[:, 0].factorize(["Besni", "Kecimen"])[0]
    )
    small1["data"]["features"] = imputer.fit_transform(small1["data"]["features"])
    selector.fit(small1["data"]["features"])
    small1["data"]["features"] = selector.transform(small1["data"]["features"])
    data["small1"] = (
        small1["data"]["features"],
        small1["data"]["targets"].iloc[:, 0].to_numpy().astype(int),
    )

    small2 = fetch_ucirepo(id=small_database_2_id)
    small2["data"]["features"] = imputer.fit_transform(small2["data"]["features"])
    selector.fit(small2["data"]["features"])
    small2["data"]["features"] = selector.transform(small2["data"]["features"])
    data["small2"] = (
        small2["data"]["features"],
        small2["data"]["targets"].iloc[:, 0].to_numpy(),
    )

    small3 = fetch_ucirepo(id=small_database_3_id)
    small3["data"]["features"] = imputer.fit_transform(small3["data"]["features"])
    selector.fit(small3["data"]["features"])
    small3["data"]["features"] = selector.transform(small3["data"]["features"])
    data["small3"] = (
        small3["data"]["features"],
        small3["data"]["targets"].iloc[:, 0].to_numpy(),
    )

    large1 = fetch_ucirepo(id=large_database_1_id)
    large1["data"]["targets"].iloc[:, 0] = (
        large1["data"]["targets"].iloc[:, 0].factorize(["b", "g"])[0]
    )
    large1["data"]["features"] = imputer.fit_transform(large1["data"]["features"])
    selector.fit(large1["data"]["features"])
    large1["data"]["features"] = selector.transform(large1["data"]["features"])
    data["large1"] = (
        large1["data"]["features"],
        large1["data"]["targets"].iloc[:, 0].to_numpy().astype(int),
    )

    large2 = fetch_ucirepo(id=large_database_2_id)
    large2["data"]["targets"].iloc[:, 0] = (
        large2["data"]["targets"].iloc[:, 0].factorize(["not fire", "fire"])[0]
    )
    large2["data"]["targets"].iloc[:, 0] = (
        large2["data"]["targets"].iloc[:, 0] < 4
    ) * 1
    large2["data"]["targets"] = large2["data"]["targets"].drop(165)
    large2["data"]["features"] = large2["data"]["features"].drop(165)
    large2["data"]["features"] = large2["data"]["features"].drop("region", axis=1)
    large2["data"]["features"]["FWI"] = large2["data"]["features"]["FWI"].astype(float)
    large2["data"]["features"]["DC"] = large2["data"]["features"]["DC"].astype(float)
    large2["data"]["features"] = imputer.fit_transform(large2["data"]["features"])
    selector.fit(large2["data"]["features"])
    large2["data"]["features"] = selector.transform(large2["data"]["features"])
    data["large2"] = (
        large2["data"]["features"],
        large2["data"]["targets"].iloc[:, 0].to_numpy().astype(int),
    )

    large3 = fetch_ucirepo(id=large_database_3_id)
    large3["data"]["targets"].iloc[:, 0] = (
        large3["data"]["targets"].iloc[:, 0].factorize(["M", "B"])[0]
    )
    large3["data"]["features"] = imputer.fit_transform(large3["data"]["features"])
    selector.fit(large3["data"]["features"])
    large3["data"]["features"] = selector.transform(large3["data"]["features"])
    data["large3"] = (
        large3["data"]["features"],
        large3["data"]["targets"].iloc[:, 0].to_numpy().astype(int),
    )

    large4 = openml.datasets.get_dataset(large_database_4_id).get_data()[0]
    large4["Class"] = large4["Class"].factorize(["1", "2"])[0]
    data["large4"] = (
        large4.loc[:, large4.columns != "Class"].to_numpy(),
        large4["Class"].to_numpy(),
    )

    large5 = openml.datasets.get_dataset(large_database_5_id).get_data()[0]
    large5["Class"] = large5["Class"].factorize(["1", "2"])[0]
    data["large5"] = (
        large5.loc[:, large5.columns != "Class"].to_numpy(),
        large5["Class"].to_numpy(),
    )

    large6 = fetch_ucirepo(id=large_database_6_id)
    large6["data"]["targets"].iloc[:, 0] = (
        large6["data"]["targets"]
        .iloc[:, 0]
        .factorize(
            ["Seker", "Barbunya", "Bombay", "Cali", "Horoz", "Dermosan", "Sira"]
        )[0]
    )
    large6["data"]["targets"].iloc[:, 0] = (
        large6["data"]["targets"].iloc[:, 0] < 5
    ) * 1
    large6["data"]["features"] = imputer.fit_transform(large6["data"]["features"])
    selector.fit(large6["data"]["features"])
    large6["data"]["features"] = selector.transform(large6["data"]["features"])
    data["large6"] = (
        large6["data"]["features"],
        large6["data"]["targets"].iloc[:, 0].to_numpy().astype(int),
    )

    return data
