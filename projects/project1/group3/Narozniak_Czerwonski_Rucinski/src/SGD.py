class SGD:
  def __init__(self, eta=0.01):
    self.eta = eta
  def update(self, t, w, b, dw, db):
    w = w - self.eta * dw
    b = b - self.eta * db
    return w, b