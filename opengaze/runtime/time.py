import time


class TimeViaEMA:
  def __init__(self, alpha: float = 0.1):
    '''Initialize the Timer with an exponential smoothing factor `alpha`.'''

    self.alpha = alpha

    self.starts = {}  # Mapping from tags to start times
    self.metric = {}  # Mapping from tags to time metric

  def tick(self, tag: str):
    '''Record the start time for a given tag.'''
    self.starts[tag] = time.perf_counter()

  def tock(self, tag: str):
    '''Record the end time for a given tag.'''
    if tag in self.starts:
      duration = time.perf_counter() - self.starts.pop(tag)
      average = self.metric.get(tag, duration)
      self.metric[tag] = (1.0 - self.alpha) * average + self.alpha * duration

  def report(self, tag: str) -> float:
    '''Return the average time for a given tag.'''
    return self.metric.get(tag, 0.0)
