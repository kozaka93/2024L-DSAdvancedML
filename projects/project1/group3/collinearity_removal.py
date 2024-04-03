import numpy as np
import heapq

def test_collinearity(columns, indices, threshold = 1e-10):
  """
  Helper function, testing whether a certain subset of columns is collinear.
  :param columns: the whole set of columns.
  :param indices: indices belonging to the subset.
  :param threshold: the value of determinant, going below which will be 
    considered being numerically collinear.
  :return: True if collinear, False otherwise.
  """
  used_columns = []
  for index in indices:
    used_columns.append(columns[index])
  
  X = np.column_stack(used_columns)
  XX = X.transpose() @ X
  if np.linalg.det(XX) < threshold:
    return True
  else:
    return False

def remove_collinear(X):
  """
  Removes the minimum number of columns to ensure the result matrix will be 
  full rank.
  :param X: a numpy matrix one needs a non-collinear version of.
  :return: a numpy matrix with collinearities removed and a set containing 
    indices of removed columns.
  """
  columns = []
  p = len(X[0])
  for i in range(p):
    columns.append(X[:,i])
  
  columns_used = []
  columns_stashed = set()
  columns_removed = set()
  for i in range(p):
    columns_used.append(i)

  heapq.heapify(columns_used)
  
  last_removed = -1
  while(True):
    if len(columns_used) == 0:
      break

    if test_collinearity(columns, columns_used):
      last_removed = heapq.heappop(columns_used)
      columns_stashed.add(last_removed)
    else:
      if last_removed == -1:
        # If the whole remaining subset is non-collinear, it's time to stop.
        
        break
      else:
        # If removing a certain column made the subset non-collinear, it means 
        # that this column is a good candidate for removal.
        
        columns_stashed.remove(last_removed)
        columns_removed.add(last_removed)
        
        # Returning stashed away columns back to the subset.
        for index in columns_stashed:
          columns_used.append(index)
        heapq.heapify(columns_used)
        columns_removed.clear()
        last_removed = -1

    # Recreating the matrix
    is_used = [False for i in range(p)]
    for index in columns_used:
      is_used[index] = True

    columns_used = []
    for i in range(p):
      if is_used[i]:
        columns_used.append(columns[i])

    X_clean = np.column_stack(columns_used)
    return X_clean, columns_removed