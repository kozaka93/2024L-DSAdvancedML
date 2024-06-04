# Custom metric
Our task slightly differs from the most common problems in ML. Our goal is not only to increase the model accuracy or any other well-known score, but our objective is to maximize profit by sufficing three conditions:

1. Each TP predicted by the model gives us 10 euro
2. Each feature costs us 200 euro
3. We may only predict 1/5 values as positive classes

In the given task we have 5000 observations and we need to select 1000 observations which we believe are positive. Thus the last condition.
